import { sanitizeUrl } from "../utils/url";
import { isBlockedDomain } from "../utils/blocklist";

interface TabSessionState {
  hasSession: boolean;
  isTracking: boolean;
  pageTitle: string;
  sourceUrl: string;
  sourceDomain: string;
  mediaType: "article" | "docs" | "video" | "pdf";
  activeDwellSeconds: number;
  estimatedCoveragePct: number;
  topKeywords: string[];
  rewindCount: number;
}

interface FlushResult {
  success: boolean;
  count: number;
  message?: string;
  error?: string;
}

// Helper to reliably find active tab across popup / browser window contexts
async function getActiveTab(): Promise<chrome.tabs.Tab | null> {
  try {
    const lastFocused = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
    if (lastFocused.length && lastFocused[0].id) return lastFocused[0];

    const currentWin = await chrome.tabs.query({ active: true, currentWindow: true });
    if (currentWin.length && currentWin[0].id) return currentWin[0];

    const anyActive = await chrome.tabs.query({ active: true });
    if (anyActive.length && anyActive[0].id) return anyActive[0];
  } catch (err) {
    console.warn("[Cognitwin Popup] Error querying active tab:", err);
  }
  return null;
}

document.addEventListener("DOMContentLoaded", async () => {
  const toggle = document.getElementById("study-mode-toggle") as HTMLInputElement;
  const modeText = document.getElementById("mode-text");
  const authAlert = document.getElementById("auth-alert");

  // Upgraded session elements
  const sessionTitleEl = document.getElementById("session-title");
  const sessionUrlEl = document.getElementById("session-url");
  const sessionBadge = document.getElementById("session-badge");
  const mediaBadge = document.getElementById("media-badge");
  const liveTimerEl = document.getElementById("live-timer");
  const liveCoverageEl = document.getElementById("live-coverage");
  const coverageBarEl = document.getElementById("coverage-bar");
  const rewindWrapper = document.getElementById("rewind-wrapper");
  const liveRewindsEl = document.getElementById("live-rewinds");
  const keywordsWrapper = document.getElementById("keywords-wrapper");
  const sessionKeywordsEl = document.getElementById("session-keywords");
  const ignoreDomainBtn = document.getElementById("ignore-domain-btn");

  // Buffer & Sync elements
  const queueCountEl = document.getElementById("queue-count");
  const syncNowBtn = document.getElementById("sync-now-btn") as HTMLButtonElement;
  const syncFeedbackEl = document.getElementById("sync-feedback");

  // Auth elements
  const tokenStatusEl = document.getElementById("token-status");
  const tokenInput = document.getElementById("token-input") as HTMLInputElement;
  const saveTokenBtn = document.getElementById("save-token-btn");
  const clearTokenBtn = document.getElementById("clear-token-btn");

  // Blocklist elements
  const toggleBlocklist = document.getElementById("toggle-blocklist");
  const blocklistPanel = document.getElementById("blocklist-panel");
  const blocklistCountEl = document.getElementById("blocklist-count");
  const blocklistItemsEl = document.getElementById("blocklist-items");
  const newDomainInput = document.getElementById("new-domain-input") as HTMLInputElement;
  const addDomainBtn = document.getElementById("add-domain-btn");

  let activeTabDomain = "";
  let activeTabId: number | null = null;
  let activeTabUrl = "";
  let localDwellSeconds = 0;
  let isTrackingActive = false;

  function formatTime(seconds: number): string {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }

  // 1. Fetch background state
  const state = await chrome.runtime.sendMessage({ type: "GET_STATE" });
  updateStateUI(state);

  // 2. Query active browser tab and query content script session state
  async function pollActiveTab(): Promise<void> {
    try {
      const activeTab = await getActiveTab();
      if (!activeTab || !activeTab.url) {
        if (sessionTitleEl) sessionTitleEl.textContent = "No active web page";
        if (sessionUrlEl) sessionUrlEl.textContent = "Open a webpage to begin tracking";
        return;
      }

      activeTabId = activeTab.id || null;
      activeTabUrl = activeTab.url || "";
      const urlInfo = sanitizeUrl(activeTabUrl);
      activeTabDomain = urlInfo.domain;

      // Extract full sanitized path without protocol
      let displayPath = urlInfo.sanitizedUrl.replace(/^https?:\/\//i, "");
      if (displayPath.length > 55) {
        displayPath = displayPath.slice(0, 52) + "...";
      }

      if (sessionUrlEl) {
        sessionUrlEl.textContent = displayPath || urlInfo.domain;
        sessionUrlEl.title = urlInfo.sanitizedUrl;
      }

      const isBlocked = isBlockedDomain(activeTabDomain, state.blocklist);

      if (urlInfo.isIgnored) {
        if (sessionBadge) {
          sessionBadge.textContent = "Ignored";
          sessionBadge.className = "pill-badge badge-ignored";
        }
        if (sessionTitleEl) sessionTitleEl.textContent = activeTab.title || urlInfo.domain;
        return;
      }

      if (isBlocked) {
        if (sessionBadge) {
          sessionBadge.textContent = "Blocked";
          sessionBadge.className = "pill-badge badge-ignored";
        }
        if (sessionTitleEl) sessionTitleEl.textContent = activeTab.title || urlInfo.domain;
        return;
      }

      // Query content script with programmatic injection fallback if not yet connected
      if (activeTabId) {
        let tabSession: TabSessionState | null = null;
        try {
          tabSession = await chrome.tabs.sendMessage(activeTabId, {
            type: "GET_PAGE_SESSION_STATE",
          });
        } catch (err: unknown) {
          const errMsg = err instanceof Error ? err.message : String(err);
          console.warn("[Cognitwin Popup] Direct message failed, attempting auto-injection:", errMsg);
          // Try auto-injecting content script if scripting permission is available
          if (chrome.scripting && activeTabId) {
            try {
              await chrome.scripting.executeScript({
                target: { tabId: activeTabId },
                files: ["content.js"],
              });
              await new Promise((r) => setTimeout(r, 200));
              tabSession = await chrome.tabs.sendMessage(activeTabId, {
                type: "GET_PAGE_SESSION_STATE",
              });
            } catch (injectErr: unknown) {
              const injectMsg = injectErr instanceof Error ? injectErr.message : String(injectErr);
              console.warn("[Cognitwin Popup] Injection retry also failed:", injectMsg);
            }
          }
        }

        if (tabSession) {
          isTrackingActive = tabSession.isTracking;
          localDwellSeconds = tabSession.activeDwellSeconds;

          if (sessionTitleEl) {
            sessionTitleEl.textContent = tabSession.pageTitle || activeTab.title || urlInfo.domain;
          }

          if (mediaBadge) {
            mediaBadge.style.display = "inline-block";
            mediaBadge.textContent = tabSession.mediaType;
            if (tabSession.mediaType === "video") {
              mediaBadge.className = "pill-badge";
              mediaBadge.style.background = "#F3E8FF";
              mediaBadge.style.color = "#7C3AED";
            } else if (tabSession.mediaType === "docs") {
              mediaBadge.className = "pill-badge";
              mediaBadge.style.background = "#EFF6FF";
              mediaBadge.style.color = "#1D4ED8";
            } else {
              mediaBadge.className = "pill-badge";
              mediaBadge.style.background = "#ECFDF5";
              mediaBadge.style.color = "#059669";
            }
          }

          if (liveTimerEl) {
            liveTimerEl.textContent = formatTime(tabSession.activeDwellSeconds);
          }

          const covPct = Math.round(tabSession.estimatedCoveragePct * 100);
          if (liveCoverageEl) {
            liveCoverageEl.textContent = `${covPct}%`;
          }
          if (coverageBarEl) {
            coverageBarEl.style.width = `${covPct}%`;
          }

          if (tabSession.mediaType === "video" && rewindWrapper && liveRewindsEl) {
            rewindWrapper.style.display = "block";
            liveRewindsEl.textContent = String(tabSession.rewindCount);
          } else if (rewindWrapper) {
            rewindWrapper.style.display = "none";
          }

          // Top 4 Keywords preview
          if (tabSession.topKeywords && tabSession.topKeywords.length > 0 && keywordsWrapper && sessionKeywordsEl) {
            keywordsWrapper.style.display = "block";
            sessionKeywordsEl.innerHTML = "";
            tabSession.topKeywords.slice(0, 4).forEach((kw) => {
              const tag = document.createElement("span");
              tag.style.background = "var(--surface)";
              tag.style.border = "1px solid var(--border-subtle)";
              tag.style.borderRadius = "4px";
              tag.style.padding = "1px 5px";
              tag.style.fontSize = "10px";
              tag.style.fontFamily = "monospace";
              tag.style.color = "var(--ink-body)";
              tag.textContent = kw;
              sessionKeywordsEl.appendChild(tag);
            });
          }

          if (sessionBadge) {
            if (state.studyMode) {
              sessionBadge.textContent = isTrackingActive ? "Tracking" : "Active";
              sessionBadge.className = "pill-badge badge-tracking";
            } else {
              sessionBadge.textContent = "Paused";
              sessionBadge.className = "pill-badge badge-paused";
            }
          }
        } else {
          // Content script not responding on this tab
          if (sessionTitleEl) sessionTitleEl.textContent = activeTab.title || urlInfo.domain;
          if (sessionBadge) {
            sessionBadge.textContent = "Disconnected";
            sessionBadge.className = "pill-badge badge-paused";
            sessionBadge.title = "Content script not responding. Refresh page (F5) to activate.";
          }
          if (syncFeedbackEl && syncFeedbackEl.style.display !== "block") {
            syncFeedbackEl.style.display = "block";
            syncFeedbackEl.style.background = "var(--fading-bg)";
            syncFeedbackEl.style.border = "1px solid var(--fading-border)";
            syncFeedbackEl.style.color = "var(--fading)";
            syncFeedbackEl.textContent = "⚠️ Tab not connected to Cognitwin. Please refresh the page (F5).";
          }
        }
      }
    } catch (err: unknown) {
      console.warn("[Cognitwin Popup] pollActiveTab error:", err);
      if (sessionTitleEl) sessionTitleEl.textContent = "Error detecting tab";
    }
  }

  await pollActiveTab();

  // Tick dwell timer locally while popup is open
  setInterval(() => {
    if (state.studyMode && isTrackingActive) {
      localDwellSeconds += 1;
      if (liveTimerEl) {
        liveTimerEl.textContent = formatTime(localDwellSeconds);
      }
    }
  }, 1000);

  // Study Mode Toggle Listener
  toggle?.addEventListener("change", async () => {
    const res = await chrome.runtime.sendMessage({ type: "TOGGLE_STUDY_MODE" });
    state.studyMode = res.studyMode === true;
    updateModeUI(state.studyMode);
    await pollActiveTab();
    const updatedState = await chrome.runtime.sendMessage({ type: "GET_STATE" });
    updateStateUI(updatedState);
  });

  function updateModeUI(isNowOn: boolean) {
    if (toggle) toggle.checked = isNowOn;
    if (modeText) {
      modeText.textContent = isNowOn ? "ON" : "OFF";
      modeText.className = isNowOn ? "mode-label on" : "mode-label";
    }
    if (sessionBadge && activeTabDomain) {
      sessionBadge.textContent = isNowOn ? (isTrackingActive ? "Tracking" : "Active") : "Paused";
      sessionBadge.className = isNowOn ? "pill-badge badge-tracking" : "pill-badge badge-paused";
    }
  }

  // Ignore current domain (1-click block)
  ignoreDomainBtn?.addEventListener("click", async () => {
    if (!activeTabDomain) return;
    const res = await chrome.runtime.sendMessage({
      type: "ADD_BLOCKLIST",
      domain: activeTabDomain,
    });
    renderBlocklist(res.blocklist || []);
    if (sessionBadge) {
      sessionBadge.textContent = "Blocked";
      sessionBadge.className = "pill-badge badge-ignored";
    }
  });

  // Save Token
  saveTokenBtn?.addEventListener("click", async () => {
    const token = tokenInput?.value?.trim();
    if (!token) return;
    await chrome.runtime.sendMessage({ type: "SET_TOKEN", token });
    tokenInput.value = "";
    const updated = await chrome.runtime.sendMessage({ type: "GET_STATE" });
    updateStateUI(updated);
  });

  // Clear Token
  clearTokenBtn?.addEventListener("click", async () => {
    await chrome.runtime.sendMessage({ type: "CLEAR_TOKEN" });
    const updated = await chrome.runtime.sendMessage({ type: "GET_STATE" });
    updateStateUI(updated);
  });

  // Sync Now with active session snapshot & feedback
  syncNowBtn?.addEventListener("click", async () => {
    if (syncNowBtn) {
      syncNowBtn.disabled = true;
      syncNowBtn.textContent = "Syncing...";
    }
    if (syncFeedbackEl) {
      syncFeedbackEl.style.display = "none";
    }

    try {
      const flushResult: FlushResult = await chrome.runtime.sendMessage({ type: "FLUSH_NOW" });

      if (syncFeedbackEl) {
        syncFeedbackEl.style.display = "block";
        if (flushResult.success) {
          syncFeedbackEl.style.background = "var(--mastered-bg)";
          syncFeedbackEl.style.border = "1px solid var(--mastered-border)";
          syncFeedbackEl.style.color = "var(--mastered)";
          syncFeedbackEl.textContent = flushResult.message || `Synced ${flushResult.count} session(s) successfully.`;
        } else {
          syncFeedbackEl.style.background = "var(--risk-bg)";
          syncFeedbackEl.style.border = "1px solid var(--risk-border)";
          syncFeedbackEl.style.color = "var(--risk)";
          syncFeedbackEl.textContent = flushResult.error || "Sync failed.";
        }
      }

      const updated = await chrome.runtime.sendMessage({ type: "GET_STATE" });
      updateStateUI(updated);
    } finally {
      if (syncNowBtn) {
        syncNowBtn.disabled = false;
        syncNowBtn.textContent = "Sync Now";
      }
    }
  });

  // Toggle Blocklist Accordion
  toggleBlocklist?.addEventListener("click", () => {
    if (!blocklistPanel) return;
    const isHidden = blocklistPanel.style.display === "none";
    blocklistPanel.style.display = isHidden ? "block" : "none";
  });

  // Add Domain to Blocklist
  addDomainBtn?.addEventListener("click", async () => {
    const domain = newDomainInput?.value?.trim();
    if (!domain) return;
    const res = await chrome.runtime.sendMessage({
      type: "ADD_BLOCKLIST",
      domain,
    });
    newDomainInput.value = "";
    renderBlocklist(res.blocklist || []);
  });

  function updateStateUI(s: {
    studyMode: boolean;
    tokenConfigured: boolean;
    authError: boolean;
    queueCount: number;
    blocklist: string[];
  }) {
    updateModeUI(s.studyMode);

    if (authAlert) {
      authAlert.className = s.authError ? "alert-banner visible" : "alert-banner";
    }

    if (queueCountEl) {
      queueCountEl.textContent = `${s.queueCount} queued`;
    }

    if (tokenStatusEl && saveTokenBtn && clearTokenBtn) {
      if (s.tokenConfigured) {
        tokenStatusEl.textContent = "Connected";
        tokenStatusEl.className = "pill-badge badge-tracking";
        saveTokenBtn.style.display = "none";
        clearTokenBtn.style.display = "inline-flex";
        if (tokenInput) tokenInput.placeholder = "Bearer Token Active (cgt_***)";
      } else {
        tokenStatusEl.textContent = "Disconnected";
        tokenStatusEl.className = "pill-badge badge-paused";
        saveTokenBtn.style.display = "inline-flex";
        clearTokenBtn.style.display = "none";
        if (tokenInput) tokenInput.placeholder = "Paste Bearer token (cgt_...)";
      }
    }

    renderBlocklist(s.blocklist || []);
  }

  function renderBlocklist(list: string[]) {
    if (blocklistCountEl) blocklistCountEl.textContent = String(list.length);
    if (!blocklistItemsEl) return;

    blocklistItemsEl.innerHTML = "";
    list.forEach((domain) => {
      const item = document.createElement("div");
      item.className = "blocklist-item";

      const name = document.createElement("span");
      name.textContent = domain;

      const removeBtn = document.createElement("span");
      removeBtn.className = "remove-domain";
      removeBtn.textContent = "×";
      removeBtn.title = "Remove domain";
      removeBtn.addEventListener("click", async () => {
        const res = await chrome.runtime.sendMessage({
          type: "REMOVE_BLOCKLIST",
          domain,
        });
        renderBlocklist(res.blocklist || []);
      });

      item.appendChild(name);
      item.appendChild(removeBtn);
      blocklistItemsEl.appendChild(item);
    });
  }
});

export {};
