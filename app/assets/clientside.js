window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        scrollSlider: function(leftClicks, rightClicks, currentValue, maxValue, windowHours) {
            // Get triggered input
            const triggered = dash_clientside.callback_context.triggered;
            if (!triggered || triggered.length === 0) {
                return window.dash_clientside.no_update;
            }
            
            const triggeredId = triggered[0].prop_id.split('.')[0];
            const step = (windowHours || 1) / 24; // Convert hours to days
            
            let newValue = currentValue || 0;
            
            if (triggeredId === 'scroll-left') {
                newValue = Math.max(0, newValue - step);
            } else if (triggeredId === 'scroll-right') {
                newValue = Math.min(maxValue || 100, newValue + step);
            }
            
            return newValue;
        }
    }
});
