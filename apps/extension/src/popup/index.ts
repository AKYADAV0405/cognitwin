// Popup script for Study Mode toggle

document.addEventListener("DOMContentLoaded", () => {
  const toggleBtn = document.getElementById("toggle-btn");
  const modeLabel = document.getElementById("mode-label");

  if (!toggleBtn || !modeLabel) return;

  chrome.storage.local.get(["studyMode"], (result) => {
    const isEnabled = Boolean(result.studyMode);
    updateUI(isEnabled);
  });

  toggleBtn.addEventListener("click", () => {
    chrome.storage.local.get(["studyMode"], (result) => {
      const nextState = !result.studyMode;
      chrome.storage.local.set({ studyMode: nextState }, () => {
        updateUI(nextState);
      });
    });
  });

  function updateUI(enabled: boolean) {
    if (!modeLabel) return;
    if (enabled) {
      modeLabel.textContent = "Study Mode: ON";
      modeLabel.className = "status-text status-active";
    } else {
      modeLabel.textContent = "Study Mode: OFF";
      modeLabel.className = "status-text";
    }
  }
});

export {};
