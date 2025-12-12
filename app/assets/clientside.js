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
            if (!metadata || !currentFigure) {
                return [window.dash_clientside.no_update, window.dash_clientside.no_update, window.dash_clientside.no_update];
            }

            const minDateTs = metadata.min_date_ts; // milliseconds
            const sliderDays = sliderValue || 0;
            const windowHrs = windowHours || 1.0;

            // Calculate start and end times
            // sliderValue is in days, convert to ms: days * 24 * 60 * 60 * 1000
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

            // Create a shallow copy of the figure to trigger a re-render in Dash/React
            const newFigure = Object.assign({}, currentFigure);
            newFigure.layout = Object.assign({}, currentFigure.layout);
            newFigure.layout.xaxis = Object.assign({}, currentFigure.layout.xaxis);

            // Update range
            newFigure.layout.xaxis.range = [startIso, endIso];

            // Format for display (DD/MM/YYYY HH:MM:SS)
            const formatDateDisplay = (d) => {
                return pad(d.getUTCDate()) + '/' +
                    pad(d.getUTCMonth() + 1) + '/' +
                    d.getUTCFullYear() + ' ' +
                    pad(d.getUTCHours()) + ':' +
                    pad(d.getUTCMinutes()) + ':' +
                    pad(d.getUTCSeconds());
            };

            return [newFigure, formatDateDisplay(startDate), formatDateDisplay(endDate)];
        }
    }
});
