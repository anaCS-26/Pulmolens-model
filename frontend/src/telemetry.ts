import { ApplicationInsights } from "@microsoft/applicationinsights-web";

const connectionString = import.meta.env.VITE_APPINSIGHTS_CONNECTION_STRING;

const appInsights = new ApplicationInsights({
  config: {
    connectionString,
    enableAutoRouteTracking: true, // Tracks SPA page views automatically
    disableAjaxTracking: false, // Tracks outgoing API calls (fetch/XHR)
    autoTrackPageVisitTime: true, // Tracks time spent on each page
    enableCorsCorrelation: true, // Correlates frontend calls with backend traces
    enableRequestHeaderTracking: true,
    enableResponseHeaderTracking: true,
  },
});

// Only load telemetry if a connection string is configured (production)
if (connectionString) {
  appInsights.loadAppInsights();
  appInsights.trackPageView(); // Initial page view
}

export { appInsights };
