console.log('=== clientside.js LOADED ===');

window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        scrollSlider: function (leftClicks, rightClicks, currentValue, maxValue, windowHours) {
            // Get triggered input
            const triggered = dash_clientside.callback_context.triggered;

            if (!triggered || triggered.length === 0) {
                return window.dash_clientside.no_update;
            }

            const triggeredId = triggered[0].prop_id.split('.')[0];

            // 25% of window size
            const windowDays = (windowHours || 1.0) / 24;
            const step = windowDays * 0.25;

            let newValue = currentValue || 0;

            if (triggeredId === 'scroll-left') {
                newValue = Math.max(0, newValue - step);
            } else if (triggeredId === 'scroll-right') {
                newValue = Math.min(maxValue || 100, newValue + step);
            }

            return newValue;
        },

        updateTimeline: function (sliderValue, windowHours, metadata, currentFigure) {
            console.log('updateTimeline called:', { sliderValue, windowHours, hasMetadata: !!metadata, hasFigure: !!currentFigure });

            if (!metadata || !currentFigure) {
                console.log('updateTimeline returning no_update - missing metadata or figure');
                return [window.dash_clientside.no_update, window.dash_clientside.no_update, window.dash_clientside.no_update];
            }

            const minDateTs = metadata.min_date_ts; // milliseconds
            const sliderDays = sliderValue || 0;
            const windowHrs = windowHours || 1.0;

            // Calculate start and end times
            const startMs = minDateTs + (sliderDays * 86400000);
            const endMs = startMs + (windowHrs * 3600000);

            const startDate = new Date(startMs);
            const endDate = new Date(endMs);

            // Format dates for Plotly (YYYY-MM-DD HH:mm:ss)
            const pad = (n) => n < 10 ? '0' + n : n;

            const formatDateISO = (d) => {
                return d.getUTCFullYear() + '-' +
                    pad(d.getUTCMonth() + 1) + '-' +
                    pad(d.getUTCDate()) + ' ' +
                    pad(d.getUTCHours()) + ':' +
                    pad(d.getUTCMinutes()) + ':' +
                    pad(d.getUTCSeconds());
            };

            const startIso = formatDateISO(startDate);
            const endIso = formatDateISO(endDate);

            console.log('updateTimeline range:', startIso, 'to', endIso);

            // Create new figure with updated range - must create new object references for React to detect change
            const newFigure = {
                data: currentFigure.data,
                layout: {
                    ...currentFigure.layout,
                    xaxis: {
                        ...currentFigure.layout.xaxis,
                        range: [startIso, endIso],
                        autorange: false
                    }
                }
            };

            console.log('Returning newFigure with range:', newFigure.layout.xaxis.range);

            // Format for display (DD/MM/YYYY HH:MM:SS)
            const formatDateDisplay = (d) => {
                return pad(d.getUTCDate()) + '/' +
                    pad(d.getUTCMonth() + 1) + '/' +
                    d.getUTCFullYear() + ' ' +
                    pad(d.getUTCHours()) + ':' +
                    pad(d.getUTCMinutes()) + ':' +
                    pad(d.getUTCSeconds());
            };

            // Return the new figure with updated range, plus the display date strings
            return [newFigure, formatDateDisplay(startDate), formatDateDisplay(endDate)];
        },

        // Navigate to previous/next event - runs entirely in browser for instant response
        navigateEvent: function (prevClicks, nextClicks, category, eventData, metadata, sliderValue, sliderMax, windowHours) {
            console.log('navigateEvent called:', { prevClicks, nextClicks, category, hasEventData: !!eventData, hasMetadata: !!metadata });

            const triggered = dash_clientside.callback_context.triggered;
            console.log('triggered:', triggered);

            if (!triggered || triggered.length === 0 || !eventData || !metadata || !category) {
                console.log('Returning no_update due to missing data');
                return window.dash_clientside.no_update;
            }

            const triggerId = triggered[0].prop_id.split('.')[0];
            const isNext = triggerId === 'next-event-btn';
            console.log('triggerId:', triggerId, 'isNext:', isNext);

            // Parse metadata
            const minDateTs = metadata.min_date_ts; // milliseconds
            console.log('minDateTs:', minDateTs);

            // Current window start time in ms
            const sliderDays = sliderValue || 0;
            const currentWindowStartMs = minDateTs + (sliderDays * 86400000);

            // Filter events by category
            const events = eventData.filter(e => e.Category === category);
            console.log('Found', events.length, 'events for category', category);

            if (events.length === 0) {
                return window.dash_clientside.no_update;
            }

            // Helper to parse date as UTC to match server-side forcing
            const parseAsUTC = (d) => {
                if (!d) return 0;
                // If it's a string and doesn't end in Z, append Z to force UTC
                if (typeof d === 'string' && !d.endsWith('Z')) {
                    return new Date(d + 'Z').getTime();
                }
                return new Date(d).getTime();
            };

            // Sort events by time
            events.sort((a, b) => parseAsUTC(a.datetime_start) - parseAsUTC(b.datetime_start));

            // Calculate window center to find next/prev relative to what we are looking at
            const windowHrs = windowHours || 1.0;
            const windowMs = windowHrs * 3600000;
            const currentCenterMs = currentWindowStartMs + (windowMs / 2);

            let targetEvent = null;

            if (isNext) {
                // Find first event starting AFTER current window center (with 1 second buffer)
                targetEvent = events.find(e => parseAsUTC(e.datetime_start) > currentCenterMs + 1000);

                // Wrap around to first event if none found
                if (!targetEvent) {
                    targetEvent = events[0];
                }
            } else {
                // Find last event starting BEFORE current window center
                const pastEvents = events.filter(e => parseAsUTC(e.datetime_start) < currentCenterMs - 1000);

                if (pastEvents.length > 0) {
                    targetEvent = pastEvents[pastEvents.length - 1];
                } else {
                    // Wrap around to last event
                    targetEvent = events[events.length - 1];
                }
            }

            if (!targetEvent) {
                return window.dash_clientside.no_update;
            }

            // Calculate new slider value to center the event in the window
            const eventStartMs = parseAsUTC(targetEvent.datetime_start);
            // Reuse windowHrs and windowMs from above
            const offsetMs = windowMs / 2;

            let newSliderDays = (eventStartMs - minDateTs - offsetMs) / 86400000;

            // Clamp to valid range
            const maxVal = sliderMax || 100;
            newSliderDays = Math.max(0, Math.min(maxVal, newSliderDays));

            console.log('Returning new slider value:', newSliderDays);
            return newSliderDays;
        }
    }
});
