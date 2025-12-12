"""
Clotheswasher Classifier - Multi-phase algorithm for identifying clotheswasher events.

This classifier uses a combination of:
1. Physical constraints (volume, duration, flow rate)
2. Temporal clustering (events within 1 hour belong to same load)
3. Flow-based grouping (clotheswashers have consistent flow rates)
4. CNN-LSTM model for final classification
"""

import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import mode
from sklearn.cluster import DBSCAN
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from src.ml.inference import CNNLSTMInference


class ClotheswasherClassifier:
    """
    Multi-phase clotheswasher event classifier.

    Phases:
    1. Toilet removal - remove very short, low-volume events
    2. Physical constraints - filter by volume, duration, flow rate
    3. Temporal clustering - group events within 1 hour as potential loads
    4. Filter small groups - remove groups with < 3 events
    5. Flow sub-clustering - sub-cluster by flow rate within temporal groups
    6. Remove evap cooler patterns - remove very regular, long-duration patterns
    7. Select candidates - select groups meeting clotheswasher criteria
    8. Group by flow - group candidates by similar flow rates
    9. CNN-LSTM scoring - score each flow group using the model
    10. Final selection - select best flow group as clotheswasher
    """

    def __init__(
        self,
        cnn_model: CNNLSTMInference,
        min_volume: float = 2.0,  # L
        max_volume: float = 200.0,  # L per event
        min_duration: float = 20.0,  # seconds
        max_duration: float = 1200.0,  # seconds (20 min)
        min_flow: float = 6.0,  # L/min
        max_flow: float = 20.0,  # L/min
        temporal_gap: float = 3600.0,  # seconds (1 hour)
        min_events_per_group: int = 3,
        flow_tolerance: float = 2.0,  # L/min for flow clustering
    ):
        self.cnn_model = cnn_model
        self.min_volume = min_volume
        self.max_volume = max_volume
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_flow = min_flow
        self.max_flow = max_flow
        self.temporal_gap = temporal_gap
        self.min_events_per_group = min_events_per_group
        self.flow_tolerance = flow_tolerance

    def classify(
        self,
        events: List[np.ndarray],
        event_features: pd.DataFrame,
        start_times: pd.DataFrame,
        verbose: bool = True,
    ) -> Dict:
        """
        Classify clotheswasher events from a list of water events.

        Args:
            events: List of raw flow series (L/10s) for each event
            event_features: DataFrame with columns: Volume, Duration, Max_flow, Mode_flow
            start_times: DataFrame with columns: year, month, day, hour, minute, second
            verbose: Print progress

        Returns:
            Dictionary with:
            - indices: List of event indices classified as clotheswasher
            - threshold_flow: Selected flow threshold
            - num_loads: Number of clotheswasher loads
            - volumes_per_load: Volume for each load
            - avg_duration: Average load duration
        """
        start_time = time.time()

        if verbose:
            print(f"\n  [Starting classification at {time.strftime('%H:%M:%S')}]")

        n_events = len(events)

        # Build datetime index
        event_datetimes = pd.to_datetime(
            start_times[["year", "month", "day", "hour", "minute", "second"]]
        )

        # Phase 1: Remove obvious toilets (very short, small events)
        phase_start = time.time()
        toilet_mask = (event_features["Volume"] < 1.0) | (
            event_features["Duration"] < 15
        )
        non_toilet_indices = np.where(~toilet_mask)[0]

        if verbose:
            print(
                f"  Phase 1 (Toilet removal): {n_events} -> {len(non_toilet_indices)} events [{time.time() - phase_start:.1f}s]"
            )

        # Phase 2: Apply physical constraints
        phase_start = time.time()

        mask = (
            (event_features["Volume"] >= self.min_volume)
            & (event_features["Volume"] <= self.max_volume)
            & (event_features["Duration"] >= self.min_duration)
            & (event_features["Duration"] <= self.max_duration)
            & (event_features["Max_flow"] >= self.min_flow)
            & (event_features["Max_flow"] <= self.max_flow)
        )
        candidate_indices = np.where(mask)[0]

        if verbose:
            print(
                f"  Phase 2 (Physical constraints): {n_events} -> {len(candidate_indices)} events [{time.time() - phase_start:.1f}s]"
            )

        if len(candidate_indices) < self.min_events_per_group:
            if verbose:
                print("  Not enough candidates after physical filtering")
            return self._empty_result()

        # Phase 3: Temporal clustering
        phase_start = time.time()

        candidate_times = event_datetimes.iloc[candidate_indices].values
        candidate_times_numeric = candidate_times.astype("datetime64[s]").astype(
            np.int64
        )

        # Sort by time
        sort_idx = np.argsort(candidate_times_numeric)
        sorted_indices = candidate_indices[sort_idx]
        sorted_times = candidate_times_numeric[sort_idx]

        # Group by temporal gaps
        temporal_groups = []
        current_group = [sorted_indices[0]]

        for i in range(1, len(sorted_indices)):
            gap = sorted_times[i] - sorted_times[i - 1]
            if gap <= self.temporal_gap:
                current_group.append(sorted_indices[i])
            else:
                temporal_groups.append(current_group)
                current_group = [sorted_indices[i]]
        temporal_groups.append(current_group)

        if verbose:
            print(
                f"  Phase 3 (Temporal clustering): {len(temporal_groups)} groups [{time.time() - phase_start:.1f}s]"
            )

        # Phase 4: Filter small groups
        phase_start = time.time()

        temporal_groups = [
            g for g in temporal_groups if len(g) >= self.min_events_per_group
        ]

        if verbose:
            print(
                f"  Phase 4 (Filter small groups): {len(temporal_groups)} groups remaining [{time.time() - phase_start:.1f}s]"
            )

        if len(temporal_groups) == 0:
            if verbose:
                print("  No temporal groups remaining")
            return self._empty_result()

        # Phase 5: Flow sub-clustering within temporal groups
        phase_start = time.time()

        flow_subgroups = []
        for group in temporal_groups:
            group_flows = event_features.loc[group, "Max_flow"].values

            if len(group) < 3:
                flow_subgroups.append(group)
                continue

            # Cluster by flow rate
            try:
                clustering = DBSCAN(eps=self.flow_tolerance, min_samples=2).fit(
                    group_flows.reshape(-1, 1)
                )
                labels = clustering.labels_

                for label in set(labels):
                    if label == -1:
                        continue
                    subgroup = [
                        group[i] for i in range(len(group)) if labels[i] == label
                    ]
                    if len(subgroup) >= 2:
                        flow_subgroups.append(subgroup)
            except:
                flow_subgroups.append(group)

        if verbose:
            print(
                f"  Phase 5 (Flow sub-clustering): {len(flow_subgroups)} subgroups [{time.time() - phase_start:.1f}s]"
            )

        # Phase 6: Remove evaporative cooler patterns
        phase_start = time.time()

        filtered_subgroups = []
        for subgroup in flow_subgroups:
            # Evap coolers run very regularly (every 15-30 min) for long periods
            if len(subgroup) > 20:
                subgroup_times = event_datetimes.iloc[subgroup].values
                subgroup_times_numeric = subgroup_times.astype("datetime64[s]").astype(
                    np.int64
                )
                gaps = np.diff(np.sort(subgroup_times_numeric))

                # Check if gaps are very regular (std < 10% of mean)
                if len(gaps) > 0:
                    mean_gap = np.mean(gaps)
                    std_gap = np.std(gaps)
                    cv = std_gap / mean_gap if mean_gap > 0 else 1

                    # Very regular pattern with short gaps = evap cooler
                    if cv < 0.15 and mean_gap < 1800:  # CV < 15%, gap < 30 min
                        continue

            filtered_subgroups.append(subgroup)

        flow_subgroups = filtered_subgroups

        if verbose:
            print(f"  Phase 6 (Remove evap cooler): {len(flow_subgroups)} subgroups")

        if len(flow_subgroups) == 0:
            if verbose:
                print("  No subgroups remaining after evap cooler removal")
            return self._empty_result()

        # Phase 7: Select candidate groups with clotheswasher characteristics
        phase_start = time.time()

        candidate_subgroups = []
        for subgroup in flow_subgroups:
            subgroup_features = event_features.loc[subgroup]

            # Clotheswasher criteria:
            # - Consistent flow rate (low CV)
            # - Typical volume per event: 5-100L
            # - Multiple events per load

            flows = subgroup_features["Max_flow"].values
            volumes = subgroup_features["Volume"].values

            flow_cv = np.std(flows) / np.mean(flows) if np.mean(flows) > 0 else 1
            avg_volume = np.mean(volumes)

            # Accept if flow is consistent and volume is reasonable
            if flow_cv < 0.3 and 3 <= avg_volume <= 150:
                candidate_subgroups.append(subgroup)

        if verbose:
            print(
                f"  Phase 7 (Select candidates): {len(candidate_subgroups)} candidates"
            )

        if len(candidate_subgroups) == 0:
            if verbose:
                print("  No candidate subgroups")
            return self._empty_result()

        # Phase 8: Group candidates by similar flow rates
        phase_start = time.time()

        # Get representative flow for each subgroup
        subgroup_flows = []
        for subgroup in candidate_subgroups:
            rep_flow = event_features.loc[subgroup, "Max_flow"].median()
            subgroup_flows.append(rep_flow)

        subgroup_flows = np.array(subgroup_flows)

        # Cluster subgroups by flow
        if len(candidate_subgroups) > 1:
            try:
                flow_clustering = DBSCAN(eps=self.flow_tolerance, min_samples=1).fit(
                    subgroup_flows.reshape(-1, 1)
                )
                flow_labels = flow_clustering.labels_
            except:
                flow_labels = np.zeros(len(candidate_subgroups))
        else:
            flow_labels = np.array([0])

        # Group subgroups by flow label
        flow_groups = defaultdict(list)
        for i, label in enumerate(flow_labels):
            flow_groups[label].append(i)

        # Build list of flow groups (each is list of subgroup indices)
        all_flow_groups = list(flow_groups.values())

        # Get representative flow for each flow group
        all_grouped_flows = []
        for group_indices in all_flow_groups:
            group_flow = np.mean([subgroup_flows[i] for i in group_indices])
            all_grouped_flows.append(group_flow)

        if verbose:
            print(f"  Phase 8 (Group by flow): {len(all_flow_groups)} flow groups")

            # Print flow groups with filtering info
            for i, (flow, group_indices) in enumerate(
                zip(all_grouped_flows, all_flow_groups)
            ):
                group_event_count = sum(
                    len(candidate_subgroups[idx]) for idx in group_indices
                )

                if flow < 6.0:
                    reason = "flow < 6 L/min"
                    if group_event_count < 5:
                        reason += f", only {group_event_count} events"
                    print(
                        f"    Flow group (FILTERED): {flow:.2f} L/min, {group_event_count} events [REMOVED - {reason}]"
                    )
                elif group_event_count < 5:
                    print(
                        f"    Flow group (FILTERED): {flow:.2f} L/min, {group_event_count} events [REMOVED - only {group_event_count} events]"
                    )
                else:
                    print(
                        f"    Flow group {i+1}: {flow:.2f} L/min, {group_event_count} events [KEPT]"
                    )

        # Filter flow groups (>= 6 L/min and >= 5 events)
        filtered_flow_groups = []
        filtered_grouped_flows = []

        for i, (flow, group_indices) in enumerate(
            zip(all_grouped_flows, all_flow_groups)
        ):
            group_event_count = sum(
                len(candidate_subgroups[idx]) for idx in group_indices
            )
            if flow >= 6.0 and group_event_count >= 5:
                filtered_flow_groups.append(group_indices)
                filtered_grouped_flows.append(flow)

        if verbose:
            print(
                f"  After filtering (flow >= 6 L/min, events >= 5): {len(filtered_flow_groups)} flow groups"
            )

        if len(filtered_flow_groups) == 0:
            if verbose:
                print("  No flow groups remaining after filtering")
            return self._empty_result()

        # Debug: Plot flow groups
        if verbose:
            self._plot_flow_groups(
                events,
                candidate_subgroups,
                all_flow_groups,
                all_grouped_flows,
                event_features,
            )

        # Phase 9: Score flow groups with CNN-LSTM
        phase_start = time.time()

        if verbose:
            print(
                f"\n  Phase 9: Running CNN-LSTM on ALL {len(all_flow_groups)} flow groups..."
            )
            print(
                f"  (Will use {len(filtered_flow_groups)} filtered groups for final selection)"
            )

        # First pass: compute CNN probs for all groups
        all_group_cw_probs = []

        for i, group_indices in enumerate(all_flow_groups):
            if verbose:
                print(f"    Processing group {i+1}/{len(all_flow_groups)}...")

            group_event_indices = []
            for subgroup_idx in group_indices:
                group_event_indices.extend(candidate_subgroups[subgroup_idx])

            # Get CNN probabilities for all events in group
            cw_probs = []
            for idx in group_event_indices:
                probs = self.cnn_model.predict(
                    raw_series=events[idx],
                    duration=event_features.loc[idx, "Duration"],
                    volume=event_features.loc[idx, "Volume"],
                    max_flow=event_features.loc[idx, "Max_flow"],
                    mode_flow=event_features.loc[idx, "Mode_flow"],
                )
                cw_probs.append(probs.get("Clotheswasher", 0.0))

            avg_cw_prob = np.mean(cw_probs)
            all_group_cw_probs.append(avg_cw_prob)

            if verbose:
                rep_flow = all_grouped_flows[i]
                print(
                    f"    Group {i+1} ({rep_flow:.2f} L/min, {len(group_event_indices)} events): Clotheswasher prob = {avg_cw_prob:.4f}"
                )

        # Calculate data duration for rate adjustment
        all_candidate_indices = []
        for subgroup in candidate_subgroups:
            all_candidate_indices.extend(subgroup)

        all_times = event_datetimes.iloc[all_candidate_indices]
        data_duration_days = (all_times.max() - all_times.min()).total_seconds() / 86400
        data_duration_weeks = max(data_duration_days / 7, 1)

        if verbose:
            print(f"\n  Data duration: {data_duration_weeks:.1f} weeks")
            print(
                f"  Expected clotheswasher: 2 loads/week × 2+ cycles/load = 4+ events/week"
            )
            min_expected_events = int(4 * data_duration_weeks)
            print(f"  Minimum expected events: {min_expected_events}")

        # Second pass: compute scores with rate adjustment
        best_combined_score = -1
        best_group_idx_in_filtered = -1
        filtered_group_counter = 0

        group_scores = []

        for i, group_indices in enumerate(all_flow_groups):
            if verbose:
                print(f"    Processing group {i+1}/{len(all_flow_groups)}...")

            group_event_indices = []
            for subgroup_idx in group_indices:
                group_event_indices.extend(candidate_subgroups[subgroup_idx])

            avg_cw_prob = all_group_cw_probs[i]
            rep_flow = all_grouped_flows[i]

            # Calculate events per week
            events_per_week = len(group_event_indices) / data_duration_weeks

            # Adjustment factor based on realistic weekly rate
            # - If events_per_week >= 4: factor = 1.0 (realistic)
            # - If events_per_week < 4: factor decreases (unlikely to be clotheswasher)
            # - If events_per_week > 20: slight penalty (might be something else)
            if events_per_week >= 4:
                if events_per_week <= 20:
                    rate_factor = 1.0  # Sweet spot
                else:
                    rate_factor = 0.9  # Slightly high, but possible
            else:
                rate_factor = events_per_week / 4.0  # Linear penalty for too few events

            # Combined score = CNN probability × exp(rate_factor)
            combined_score = avg_cw_prob * math.exp(rate_factor)
            group_scores.append(combined_score)

            if verbose:
                print(
                    f"    Group {i+1} ({rep_flow:.2f} L/min, {len(group_event_indices)} events): "
                    f"CW prob={avg_cw_prob:.4f}, rate={events_per_week:.1f}/wk, "
                    f"rate_factor={rate_factor:.2f}, score={combined_score:.4f}"
                )

            # Check if this group passed the filter (>= 6 L/min and >= 5 events)
            if rep_flow >= 6.0 and len(group_event_indices) >= 5:
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_group_idx_in_filtered = filtered_group_counter
                filtered_group_counter += 1

        # Print summary of all groups
        if verbose:
            print(f"\n  SUMMARY - All {len(all_group_cw_probs)} flow groups:")
            for i, (rep_flow, cw_prob, score) in enumerate(
                zip(all_grouped_flows, all_group_cw_probs, group_scores)
            ):
                group_events_count = sum(
                    len(candidate_subgroups[idx]) for idx in all_flow_groups[i]
                )
                events_per_week = group_events_count / data_duration_weeks
                status = (
                    "FILTERED" if (rep_flow < 6.0 or group_events_count < 5) else "KEPT"
                )
                print(
                    f"    Group {i+1}: {rep_flow:.2f} L/min, {group_events_count} events ({events_per_week:.1f}/wk), "
                    f"CW prob={cw_prob:.4f}, score={score:.4f} [{status}]"
                )

        if best_group_idx_in_filtered == -1:
            if verbose:
                print("  No group selected")
            return self._empty_result()

        if verbose:
            print(
                f"\n  Selected Group (from filtered): {filtered_grouped_flows[best_group_idx_in_filtered]:.2f} L/min "
                f"(Combined score: {best_combined_score:.4f}) [{time.time() - phase_start:.1f}s]"
            )

        # Phase 10: Use selected flow group as threshold
        phase_start = time.time()

        selected_flow = filtered_grouped_flows[best_group_idx_in_filtered]

        if verbose:
            print(f"\n  Phase 10: Using flow threshold = {selected_flow:.2f} L/min")

        # Get all events within flow tolerance of selected flow
        flow_mask = (
            np.abs(event_features["Max_flow"] - selected_flow) <= self.flow_tolerance
        )

        # Also apply basic physical constraints
        combined_mask = (
            flow_mask
            & (event_features["Volume"] >= self.min_volume)
            & (event_features["Volume"] <= self.max_volume)
            & (event_features["Duration"] >= self.min_duration)
            & (event_features["Duration"] <= self.max_duration)
        )

        cw_indices = np.where(combined_mask)[0].tolist()

        if verbose:
            print(
                f"  Found {len(cw_indices)} clotheswasher events [{time.time() - phase_start:.1f}s]"
            )

        # Temporal refinement: group into loads
        num_loads, volumes_per_load, avg_duration = self._group_into_loads(
            cw_indices, event_features, event_datetimes
        )

        if verbose:
            print(
                f"  Temporal refinement complete: {num_loads} loads [{time.time() - phase_start:.1f}s]"
            )
            print(
                f"\n[{time.strftime('%H:%M:%S')}] Classification completed in {time.time() - start_time:.2f} seconds"
            )

        return {
            "indices": cw_indices,
            "threshold_flow": selected_flow,
            "num_loads": num_loads,
            "volumes_per_load": volumes_per_load,
            "avg_duration": avg_duration,
        }

    def _empty_result(self) -> Dict:
        """Return empty result dictionary."""
        return {
            "indices": [],
            "threshold_flow": 0.0,
            "num_loads": 0,
            "volumes_per_load": [],
            "avg_duration": 0.0,
        }

    def _group_into_loads(
        self,
        indices: List[int],
        event_features: pd.DataFrame,
        event_datetimes: pd.Series,
    ) -> Tuple[int, List[float], float]:
        """Group clotheswasher events into loads based on temporal proximity."""
        if len(indices) == 0:
            return 0, [], 0.0

        # Sort by time
        times = event_datetimes.iloc[indices].values
        times_numeric = times.astype("datetime64[s]").astype(np.int64)
        sort_idx = np.argsort(times_numeric)
        sorted_indices = [indices[i] for i in sort_idx]
        sorted_times = times_numeric[sort_idx]

        # Group by temporal gaps (1 hour)
        loads = []
        current_load = [sorted_indices[0]]

        for i in range(1, len(sorted_indices)):
            gap = sorted_times[i] - sorted_times[i - 1]
            if gap <= self.temporal_gap:
                current_load.append(sorted_indices[i])
            else:
                loads.append(current_load)
                current_load = [sorted_indices[i]]
        loads.append(current_load)

        # Calculate volumes per load
        volumes_per_load = []
        durations = []

        for load in loads:
            load_volume = event_features.loc[load, "Volume"].sum()
            volumes_per_load.append(load_volume)

            load_times = event_datetimes.iloc[load]
            load_duration = (load_times.max() - load_times.min()).total_seconds()
            durations.append(load_duration)

        avg_duration = np.mean(durations) if durations else 0.0

        return len(loads), volumes_per_load, avg_duration

    def _plot_flow_groups(
        self,
        events: List[np.ndarray],
        candidate_subgroups: List[List[int]],
        all_flow_groups: List[List[int]],
        all_grouped_flows: List[float],
        event_features: pd.DataFrame,
    ):
        """Plot sample events from each flow group for debugging."""
        n_groups = min(len(all_flow_groups), 6)  # Max 6 groups

        if n_groups == 0:
            return

        print(f"\n  DEBUG: Plotting {n_groups} flow groups...")

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for i in range(n_groups):
            ax = axes[i]
            group_indices = all_flow_groups[i]
            rep_flow = all_grouped_flows[i]

            # Get all event indices in this group
            event_indices = []
            for subgroup_idx in group_indices:
                event_indices.extend(candidate_subgroups[subgroup_idx])

            # Plot first few events
            n_to_plot = min(5, len(event_indices))
            for j in range(n_to_plot):
                idx = event_indices[j]
                series = events[idx] * 6  # Convert to L/min
                time_seconds = np.arange(len(series)) * 10
                ax.plot(time_seconds, series, alpha=0.7, linewidth=1)

            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Flow Rate (L/min)")
            ax.set_title(
                f"Group {i+1}: {rep_flow:.1f} L/min\n({len(event_indices)} events)"
            )
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for i in range(n_groups, 6):
            axes[i].axis("off")

        plt.tight_layout()
        plt.savefig("flow_groups_debug.png", dpi=150, bbox_inches="tight")
        print(f"  Saved flow groups plot to: flow_groups_debug.png")
        plt.close()
