import { sanitizeUrl } from "../utils/url";
import { isBlockedDomain } from "../utils/blocklist";

document.addEventListener("DOMContentLoaded", async () => {
  const toggle = document.getElementById("study-mode-toggle") as HTMLInputElement;
  const modeText = document.getElementById("mode-text");
  const authAlert = document.getElementById("auth-alert");
  const currentDomainEl = document.getElementById("current-domain");
  const sessionBadge = document.getElementById("session-badge");
  const ignoreDomainBtn = document.getElementById("ignore-domain-btn");
  const queueCountEl = document.getElementById("queue-count");
  const syncNowBtn = document.getElementById("sync-now-btn");
  const tokenStatusEl = document.getElementById("token-status");
  const tokenInput = document.getElementById("token-input") as HTMLInputElement;
  const saveTokenBtn = document.getElementById("save-token-btn");
  const clearTokenBtn = document.getElementById("clear-token-btn");
  const toggleBlocklist = document.getElementById("toggle-blocklist");
  const blocklistPanel = document.getElementById("blocklist-panel");
  const blocklistCountEl = document.getElementById("blocklist-count");
  const blocklistItemsEl = document.getElementById("blocklist-items");
  const newDomainInput = document.getElementById("new-domain-input") as HTMLInputElement;
  const addDomainBtn = document.getElementById("add-domain-btn");

  let activeTabDomain = "";

  // 1. Fetch background state
  const state = await chrome.runtime.sendMessage({ type: "GET_STATE" });
  updateStateUI(state);

  // 2. Query active browser tab
  try {
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tabs.length && tabs[0].url) {
      const urlInfo = sanitizeUrl(tabs[0].url);
      activeTabDomain = urlInfo.domain;
      if (currentDomainEl) {
        currentDomainEl.textContent = activeTabDomain || "New Tab";
      }

      const isBlocked = isBlockedDomain(activeTabDomain, state.blocklist);
      if (sessionBadge) {
        if (urlInfo.isIgnored) {
          sessionBadge.textContent = "Ignored";
          sessionBadge.className = "pill-badge badge-ignored";
        } else if (isBlocked) {
          sessionBadge.textContent = "Blocked";
          sessionBadge.className = "pill-badge badge-ignored";
        } else if (state.studyMode) {
          sessionBadge.textContent = "Tracking";
          sessionBadge.className = "pill-badge badge-tracking";
        } else {
          sessionBadge.textContent = "Paused";
          sessionBadge.className = "pill-badge badge-paused";
        }
      }
    }
  } catch {
    if (currentDomainEl) currentDomainEl.textContent = "No active tab";
  }

  // Study Mode Toggle Listener
  toggle?.addEventListener("change", async () => {
    const res = await chrome.runtime.sendMessage({ type: "TOGGLE_STUDY_MODE" });
    const isNowOn = res.studyMode === true;
    if (modeText) {
      modeText.textContent = isNowOn ? "ON" : "OFF";
      modeText.className = isNowOn ? "mode-label on" : "mode-label";
    }
    if (sessionBadge && activeTabDomain) {
      sessionBadge.textContent = isNowOn ? "Tracking" : "Paused";
      sessionBadge.className = isNowOn ? "pill-badge badge-tracking" : "pill-badge badge-paused";
    }
  });

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

  // Sync Now
  syncNowBtn?.addEventListener("click", async () => {
    if (syncNowBtn) syncNowBtn.textContent = "Syncing...";
    await chrome.runtime.sendMessage({ type: "FLUSH_NOW" });
    const updated = await chrome.runtime.sendMessage({ type: "GET_STATE" });
    updateStateUI(updated);
    if (syncNowBtn) syncNowBtn.textContent = "Sync Now";
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
    if (toggle) toggle.checked = s.studyMode;
    if (modeText) {
      modeText.textContent = s.studyMode ? "ON" : "OFF";
      modeText.className = s.studyMode ? "mode-label on" : "mode-label";
    }

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
