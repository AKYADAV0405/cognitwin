// Background Service Worker for Cognitwin Manifest V3 Extension
// Enforces passive focus monitoring, active tab idle detection, and batched sync
export const initialSessionState = {
    studyModeEnabled: false,
    activeDomain: null,
    dwellSeconds: 0,
};
// Listen for lifecycle events
chrome.runtime.onInstalled.addListener(() => {
    console.log("[Cognitwin] Extension installed successfully with state:", initialSessionState);
    chrome.storage.local.set({ studyMode: false, token: null });
});
