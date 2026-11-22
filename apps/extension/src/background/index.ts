import { DEFAULT_PRIVACY_BLOCKLIST, isBlockedDomain } from "../utils/blocklist";
import { checkRevisit, updateRevisitCache, RevisitCache } from "../utils/revisit";
import { coalesceTelemetryEvents, processSleepGap, RawTelemetryEvent } from "../utils/coalesce";
import { sanitizeUrl, isPdfUrl } from "../utils/url";
import { cleanPageTitle } from "../utils/nlp";
import { createPdfSession, calculatePdfCoverage, PdfSessionState } from "../utils/pdf";

const ALARM_NAME = "sync_passive_telemetry";
const MAX_QUEUE_SIZE = 200;
const BATCH_FLUSH_SIZE = 20;

let lastCheckpointMs = Date.now();
let isSystemActive = true;

// Active background PDF session state
let activePdfSession: PdfSessionState | null = null;
let isPdfTabFocused = false;

// Helper to reliably find active tab across service worker contexts
async function getActiveTab(): Promise<chrome.tabs.Tab | null> {
  try {
    const lastFocused = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
    if (lastFocused.length && lastFocused[0].id) return lastFocused[0];

    const currentWin = await chrome.tabs.query({ active: true, currentWindow: true });
    if (currentWin.length && currentWin[0].id) return currentWin[0];

    const anyActive = await chrome.tabs.query({ active: true });
    if (anyActive.length && anyActive[0].id) return anyActive[0];
  } catch (err) {
    console.warn("[Cognitwin Background] Error querying active tab:", err);
  }
  return null;
}

// Initialize on installed / reloaded
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

  // Auto-inject content script into all open tabs on extension install / reload
  try {
    const openTabs = await chrome.tabs.query({ url: ["http://*/*", "https://*/*"] });
    for (const tab of openTabs) {
      if (tab.id && chrome.scripting && !isPdfUrl(tab.url || "", tab.title)) {
        chrome.scripting.executeScript({
          target: { tabId: tab.id },
          files: ["content.js"],
        }).catch(() => {});
      }
    }
  } catch {
    // Ignore permissions or restricted tabs
  }
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

// -------------------------------------------------------------
// PDF BACKGROUND TRACKING LIFECYCLE
// -------------------------------------------------------------
async function checkAndActivatePdfTab(tabId?: number): Promise<void> {
  try {
    let tab: chrome.tabs.Tab | null = null;
    if (tabId) {
      tab = await chrome.tabs.get(tabId).catch(() => null);
    } else {
      tab = await getActiveTab();
    }

    if (!tab || !tab.url) {
      isPdfTabFocused = false;
      return;
    }

    const isPdf = isPdfUrl(tab.url, tab.title);
    if (!isPdf) {
      isPdfTabFocused = false;
      return;
    }

    const urlInfo = sanitizeUrl(tab.url);
    if (urlInfo.isIgnored) {
      isPdfTabFocused = false;
      return;
    }

    const storage = await chrome.storage.local.get(["studyMode", "blocklist"]);
    if (isBlockedDomain(urlInfo.domain, storage.blocklist || DEFAULT_PRIVACY_BLOCKLIST)) {
      isPdfTabFocused = false;
      return;
    }

    isPdfTabFocused = true;
    const now = Date.now();

    if (activePdfSession && activePdfSession.sanitizedUrl === urlInfo.sanitizedUrl) {
      // Resume existing PDF session
      activePdfSession.tabId = tab.id || activePdfSession.tabId;
      activePdfSession.lastHeartbeatMs = now;
      if (tab.title) {
        activePdfSession.pageTitle = cleanPageTitle(tab.title);
      }
    } else {
      // Flush previous PDF session if it accumulated dwell
      if (activePdfSession) {
        await syncActivePdfSession();
      }
      activePdfSession = createPdfSession(
        tab.id || 0,
        urlInfo.sanitizedUrl,
        urlInfo.domain,
        tab.title,
        new Date().toISOString(),
        now
      );
    }
  } catch (err) {
    console.warn("[Cognitwin Background] checkAndActivatePdfTab error:", err);
  }
}

async function syncActivePdfSession(): Promise<{ flushed: boolean; message?: string }> {
  if (!activePdfSession) {
    return { flushed: false, message: "No active PDF session." };
  }
  const unsyncedDwell = activePdfSession.activeDwellSeconds - activePdfSession.syncedDwellSeconds;
  if (unsyncedDwell < 15) {
    const msg =
      activePdfSession.syncedDwellSeconds > 0
        ? `Need at least 15s of new active study time before syncing PDF (already synced: ${activePdfSession.syncedDwellSeconds}s, current total: ${activePdfSession.activeDwellSeconds}s).`
        : `Need at least 15s of active study time before syncing PDF (current: ${activePdfSession.activeDwellSeconds}s).`;
    return { flushed: false, message: msg };
  }

  const cov = calculatePdfCoverage(activePdfSession.activeDwellSeconds);
  const pdfEvent: RawTelemetryEvent = {
    client_event_id:
      typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `pdf-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    source_domain: activePdfSession.sourceDomain,
    source_url: activePdfSession.sanitizedUrl,
    page_title: activePdfSession.pageTitle,
    meta_description: `PDF Document: ${activePdfSession.pageTitle}`.slice(0, 200),
    media_type: "pdf",
    active_dwell_seconds: unsyncedDwell,
    estimated_coverage_pct: cov,
    viewed_headings: [activePdfSession.pageTitle],
    top_keywords: activePdfSession.topKeywords,
    is_revisit: false,
    days_since_last_visit: null,
    started_at: activePdfSession.startedAt,
    ended_at: new Date().toISOString(),
  };

  activePdfSession.syncedDwellSeconds = activePdfSession.activeDwellSeconds;
  await handleIncomingTelemetry(pdfEvent);
  return { flushed: true };
}

// Listen to tab activation (tab switch)
chrome.tabs.onActivated.addListener(async (activeInfo) => {
  await checkAndActivatePdfTab(activeInfo.tabId);
});

// Listen to window focus changed
chrome.windows.onFocusChanged.addListener(async (windowId) => {
  if (windowId === chrome.windows.WINDOW_ID_NONE) {
    isPdfTabFocused = false;
  } else {
    await checkAndActivatePdfTab();
  }
});

// Listen to tab URL updates
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo) => {
  if (changeInfo.url || changeInfo.title || changeInfo.status === "complete") {
    const active = await getActiveTab();
    if (active && active.id === tabId) {
      await checkAndActivatePdfTab(tabId);
    }
  }
});

// Listen to tab closing
chrome.tabs.onRemoved.addListener(async (tabId) => {
  if (activePdfSession && activePdfSession.tabId === tabId) {
    isPdfTabFocused = false;
    await syncActivePdfSession();
    activePdfSession = null;
  }
});

// 1-second PDF dwell accumulator heartbeat
setInterval(async () => {
  if (!activePdfSession || !isPdfTabFocused || !isSystemActive) {
    return;
  }
  const storage = await chrome.storage.local.get(["studyMode"]);
  if (storage.studyMode !== true) {
    return;
  }

  const now = Date.now();
  const deltaSec = Math.round((now - activePdfSession.lastHeartbeatMs) / 1000);
  activePdfSession.lastHeartbeatMs = now;

  // Sleep-gap cap: discard sleep gaps > 60s
  if (deltaSec > 60 || deltaSec < 0) {
    return;
  }

  if (deltaSec > 0 && deltaSec <= 3) {
    activePdfSession.activeDwellSeconds += deltaSec;
  }
}, 1000);

// -------------------------------------------------------------
// MAIN MESSAGE DISPATCHER
// -------------------------------------------------------------
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

  if (message.type === "GET_BACKGROUND_PDF_SESSION") {
    if (activePdfSession && (message.url ? activePdfSession.sanitizedUrl === message.url : true)) {
      chrome.storage.local.get(["studyMode"]).then((storage) => {
        const isStudyOn = storage.studyMode === true;
        const cov = calculatePdfCoverage(activePdfSession ? activePdfSession.activeDwellSeconds : 0);
        sendResponse({
          hasSession: true,
          isTracking: isStudyOn && isPdfTabFocused,
          pageTitle: activePdfSession ? activePdfSession.pageTitle : "PDF Document",
          sourceUrl: activePdfSession ? activePdfSession.sanitizedUrl : "",
          sourceDomain: activePdfSession ? activePdfSession.sourceDomain : "",
          mediaType: "pdf",
          activeDwellSeconds: activePdfSession ? activePdfSession.activeDwellSeconds : 0,
          estimatedCoveragePct: cov,
          topKeywords: activePdfSession ? activePdfSession.topKeywords.slice(0, 4) : [],
          rewindCount: 0,
        });
      });
    } else {
      sendResponse({ hasSession: false });
    }
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
      if (tab.id && !isPdfUrl(tab.url || "", tab.title)) {
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
    return;
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

export interface FlushResult {
  success: boolean;
  count: number;
  message?: string;
  error?: string;
}

async function flushEventQueue(_reason: string): Promise<FlushResult> {
  // 1. First trigger snapshot on active tab (or active PDF session)
  let activeTabMessage: string | null = null;
  let tabErrorReason: string | null = null;
  try {
    const activeTab = await getActiveTab();
    if (activeTab && isPdfUrl(activeTab.url || "", activeTab.title)) {
      // PDF tab: snapshot and sync background PDF session
      const pdfSync = await syncActivePdfSession();
      if (!pdfSync.flushed && pdfSync.message) {
        activeTabMessage = pdfSync.message;
      }
    } else if (activeTab && activeTab.id) {
      try {
        const res = await chrome.tabs.sendMessage(activeTab.id, { type: "REQUEST_SESSION_FLUSH" });
        if (res) {
          if (!res.flushed && res.message) {
            activeTabMessage = res.message;
          } else if (res.flushed && res.event) {
            await handleIncomingTelemetry(res.event);
          }
        } else {
          tabErrorReason = "Tab returned empty response";
        }
      } catch (err: unknown) {
        tabErrorReason = err instanceof Error ? err.message : String(err);
        console.warn("[Cognitwin Background] REQUEST_SESSION_FLUSH failed on tab", activeTab.id, tabErrorReason);
      }
    } else {
      tabErrorReason = "No active web tab detected";
    }
  } catch (err: unknown) {
    tabErrorReason = err instanceof Error ? err.message : String(err);
  }

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
    if (activeTabMessage) {
      return { success: false, count: 0, error: activeTabMessage };
    }
    if (tabErrorReason) {
      return {
        success: false,
        count: 0,
        error: `Cannot sync active tab (${tabErrorReason}). Please refresh the page (F5).`,
      };
    }
    return { success: true, count: 0, message: "Queue is empty. No sessions to sync." };
  }

  if (!token) {
    return { success: false, count: 0, error: "No Bearer token configured. Please connect your token below." };
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
      const data = await res.json().catch(() => ({ ingested: batch.length, duplicates_ignored: 0 }));
      const remainingQueue = queue.slice(batch.length);
      await chrome.storage.local.set({
        eventQueue: remainingQueue,
        authError: false,
      });
      return {
        success: true,
        count: data.ingested,
        message: `Synced ${data.ingested} session(s) successfully (${data.duplicates_ignored} deduplicated).`,
      };
    }

    if (res.status === 401) {
      await chrome.storage.local.set({ authError: true });
      return { success: false, count: 0, error: "401 Unauthorized: Invalid or revoked Bearer token." };
    }

    if (res.status === 422) {
      const err = await res.json().catch(() => ({}));
      return { success: false, count: 0, error: `422 Validation Error: ${JSON.stringify(err.detail || err)}` };
    }

    return { success: false, count: 0, error: `HTTP ${res.status}: ${res.statusText}` };
  } catch {
    return { success: false, count: 0, error: `API unreachable at ${apiUrl}. Is backend running?` };
  }
}
