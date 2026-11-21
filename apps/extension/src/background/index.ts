import { DEFAULT_PRIVACY_BLOCKLIST } from "../utils/blocklist";
import { checkRevisit, updateRevisitCache, RevisitCache } from "../utils/revisit";
import { coalesceTelemetryEvents, processSleepGap, RawTelemetryEvent } from "../utils/coalesce";

const ALARM_NAME = "sync_passive_telemetry";
const MAX_QUEUE_SIZE = 200;
const BATCH_FLUSH_SIZE = 20;

let lastCheckpointMs = Date.now();
let isSystemActive = true;

// Initialize on installed
chrome.runtime.onInstalled.addListener(async () => {
  const existing = await chrome.storage.local.get(["studyMode", "blocklist", "apiUrl", "revisitCache", "eventQueue"]);

  const updates: Record<string, unknown> = {};
  if (typeof existing.studyMode === "undefined") {
    updates.studyMode = false;
  }
  if (!existing.blocklist) {
    updates.blocklist = DEFAULT_PRIVACY_BLOCKLIST;
  }
  if (!existing.apiUrl) {
    updates.apiUrl = "http://localhost:8000";
  }
  if (!existing.revisitCache) {
    updates.revisitCache = {};
  }
  if (!existing.eventQueue) {
    updates.eventQueue = [];
  }

  if (Object.keys(updates).length > 0) {
    await chrome.storage.local.set(updates);
  }

  // Periodic flush alarm (every 3 minutes)
  chrome.alarms.create(ALARM_NAME, { periodInMinutes: 3 });
  updateBadge(existing.studyMode === true);
});

// Update extension icon badge
function updateBadge(isStudyModeOn: boolean): void {
  if (isStudyModeOn) {
    chrome.action.setBadgeText({ text: "ON" });
    chrome.action.setBadgeBackgroundColor({ color: "#059669" }); // Emerald
  } else {
    chrome.action.setBadgeText({ text: "" });
  }
}

// Track idle state (60s threshold)
chrome.idle.setDetectionInterval(60);
chrome.idle.onStateChanged.addListener((newState) => {
  isSystemActive = newState === "active";
});

// Periodic alarm listener
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === ALARM_NAME) {
    flushEventQueue("alarm_3min");
  }
});

// Main message dispatcher
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === "FLUSH_TELEMETRY_EVENT") {
    handleIncomingTelemetry(message.event).then(() => {
      sendResponse({ status: "queued" });
    });
    return true;
  }

  if (message.type === "GET_STATE") {
    getState().then((state) => sendResponse(state));
    return true;
  }

  if (message.type === "TOGGLE_STUDY_MODE") {
    toggleStudyMode().then((newMode) => sendResponse({ studyMode: newMode }));
    return true;
  }

  if (message.type === "SET_TOKEN") {
    chrome.storage.local.set({ extensionToken: message.token, authError: false }).then(() => {
      sendResponse({ success: true });
    });
    return true;
  }

  if (message.type === "CLEAR_TOKEN") {
    chrome.storage.local.remove(["extensionToken"]).then(() => {
      sendResponse({ success: true });
    });
    return true;
  }

  if (message.type === "ADD_BLOCKLIST") {
    addToBlocklist(message.domain).then((list) => sendResponse({ blocklist: list }));
    return true;
  }

  if (message.type === "REMOVE_BLOCKLIST") {
    removeFromBlocklist(message.domain).then((list) => sendResponse({ blocklist: list }));
    return true;
  }

  if (message.type === "FLUSH_NOW") {
    flushEventQueue("manual_popup").then((result) => sendResponse(result));
    return true;
  }

  return false;
});

async function getState() {
  const storage = await chrome.storage.local.get([
    "studyMode",
    "extensionToken",
    "apiUrl",
    "authError",
    "blocklist",
    "eventQueue",
  ]);

  return {
    studyMode: storage.studyMode === true,
    tokenConfigured: !!storage.extensionToken,
    authError: storage.authError === true,
    apiUrl: storage.apiUrl || "http://localhost:8000",
    queueCount: (storage.eventQueue || []).length,
    blocklist: storage.blocklist || DEFAULT_PRIVACY_BLOCKLIST,
  };
}

async function toggleStudyMode(): Promise<boolean> {
  const storage = await chrome.storage.local.get(["studyMode"]);
  const nextMode = !storage.studyMode;

  await chrome.storage.local.set({ studyMode: nextMode });
  updateBadge(nextMode);

  // Notify all open tabs of study mode change
  try {
    const tabs = await chrome.tabs.query({});
    for (const tab of tabs) {
      if (tab.id) {
        chrome.tabs.sendMessage(tab.id, {
          type: "STUDY_MODE_CHANGED",
          studyMode: nextMode,
        }).catch(() => {
          // Tab may not have content script injected
        });
      }
    }
  } catch {
    // Ignore tab query errors
  }

  if (!nextMode) {
    // Immediately flush on Study Mode OFF
    await flushEventQueue("study_mode_off");
  }

  return nextMode;
}

async function addToBlocklist(domain: string): Promise<string[]> {
  const clean = domain.toLowerCase().trim();
  if (!clean) return [];

  const storage = await chrome.storage.local.get(["blocklist"]);
  const current: string[] = storage.blocklist || DEFAULT_PRIVACY_BLOCKLIST;
  if (!current.includes(clean)) {
    const updated = [...current, clean];
    await chrome.storage.local.set({ blocklist: updated });
    return updated;
  }
  return current;
}

async function removeFromBlocklist(domain: string): Promise<string[]> {
  const clean = domain.toLowerCase().trim();
  const storage = await chrome.storage.local.get(["blocklist"]);
  const current: string[] = storage.blocklist || DEFAULT_PRIVACY_BLOCKLIST;
  const updated = current.filter((d) => d !== clean);
  await chrome.storage.local.set({ blocklist: updated });
  return updated;
}

async function handleIncomingTelemetry(event: RawTelemetryEvent): Promise<void> {
  if (!isSystemActive) {
    return;
  }
  const now = Date.now();

  // 1. Process sleep-gap delta
  const { isSleepGap } = processSleepGap(lastCheckpointMs, now);
  lastCheckpointMs = now;

  if (isSleepGap) {
    return; // Discard event generated across OS sleep
  }

  // 2. Revisit calculation and cache update
  const storage = await chrome.storage.local.get(["revisitCache", "eventQueue"]);
  const revisitCache: RevisitCache = storage.revisitCache || {};

  const { isRevisit, daysSinceLastVisit } = checkRevisit(revisitCache, event.source_url, now);
  event.is_revisit = isRevisit;
  event.days_since_last_visit = daysSinceLastVisit;

  const updatedRevisitCache = updateRevisitCache(
    revisitCache,
    event.source_url,
    event.estimated_coverage_pct,
    now
  );

  // 3. Queue management with coalescing
  const rawQueue: RawTelemetryEvent[] = storage.eventQueue || [];
  rawQueue.push(event);

  // Coalesce events
  let coalesced = coalesceTelemetryEvents(rawQueue);

  // Cap at 200 items (FIFO eviction if oversized)
  if (coalesced.length > MAX_QUEUE_SIZE) {
    coalesced = coalesced.slice(coalesced.length - MAX_QUEUE_SIZE);
  }

  await chrome.storage.local.set({
    revisitCache: updatedRevisitCache,
    eventQueue: coalesced,
  });

  // Flush early if queue reaches 10 items
  if (coalesced.length >= 10) {
    await flushEventQueue("queue_threshold_10");
  }
}

async function flushEventQueue(_reason: string): Promise<{ success: boolean; count: number }> {
  const storage = await chrome.storage.local.get([
    "eventQueue",
    "extensionToken",
    "apiUrl",
    "authError",
  ]);

  const queue: RawTelemetryEvent[] = storage.eventQueue || [];
  const token: string | undefined = storage.extensionToken;
  const apiUrl: string = storage.apiUrl || "http://localhost:8000";

  if (!queue.length) {
    return { success: true, count: 0 };
  }

  if (!token) {
    // If no token, keep events in offline queue
    return { success: false, count: 0 };
  }

  // Drain in chunks of up to 20 events
  const batch = queue.slice(0, BATCH_FLUSH_SIZE);

  try {
    const res = await fetch(`${apiUrl}/api/v1/telemetry/passive`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ events: batch }),
    });

    if (res.status === 202) {
      // Success: remove sent batch from queue
      const remainingQueue = queue.slice(batch.length);
      await chrome.storage.local.set({
        eventQueue: remainingQueue,
        authError: false,
      });
      return { success: true, count: batch.length };
    }

    if (res.status === 401) {
      // Token revoked or invalid: pause flushing without dropping queue
      await chrome.storage.local.set({ authError: true });
      return { success: false, count: 0 };
    }

    return { success: false, count: 0 };
  } catch {
    // Network offline / server down: keep events in local storage for retry
    return { success: false, count: 0 };
  }
}
