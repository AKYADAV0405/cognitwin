console.info("[Cognitwin] content script active on", location.hostname, location.href);

import { sanitizeUrl } from "../utils/url";
import { isBlockedDomain } from "../utils/blocklist";
import { calculateArticleCoverage } from "../utils/coverage";
import { extractKeywords, extractHeadings, extractKeywordsFromFilename } from "../utils/nlp";
import { calculateVideoCoverage, isRewindSeek, getBucketIndex } from "../utils/video";

interface ContentSession {
  clientEventId: string;
  sourceUrl: string;
  sourceDomain: string;
  pageTitle: string;
  mediaType: "article" | "docs" | "video" | "pdf";
  startedAt: string;
  totalWords: number;
  viewedWords: number;
  viewedHeadings: string[];
  topKeywords: string[];
  // Video fields
  videoDurationSec?: number;
  channelName?: string;
  watchedBuckets: Set<number>;
  rewindCount: number;
  pauseCount: number;
  lastVideoTime: number;
  playbackRates: number[];
}

let currentSession: ContentSession | null = null;
let activeDwellSeconds = 0;
let lastHeartbeatTime = Date.now();
let isTabVisible = document.visibilityState === "visible";
let isStudyModeOn = false;

// Generate UUIDv4 client side
function generateUUID(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

// -------------------------------------------------------------
// TOP-LEVEL MESSAGE DISPATCHER (Registered synchronously)
// -------------------------------------------------------------
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  console.info("[Cognitwin] Content script received message:", message?.type);

  if (message.type === "PING") {
    sendResponse({
      pong: true,
      href: window.location.href,
      studyMode: isStudyModeOn,
      hasSession: !!currentSession,
    });
    return true;
  }

  if (message.type === "STUDY_MODE_CHANGED") {
    const wasStudyMode = isStudyModeOn;
    isStudyModeOn = message.studyMode === true;
    console.info("[Cognitwin] Study mode changed to:", isStudyModeOn);
    if (wasStudyMode && !isStudyModeOn) {
      flushSession("study_mode_off", true);
    } else if (!wasStudyMode && isStudyModeOn) {
      if (!currentSession) {
        initContentTracker(true);
      }
    }
    sendResponse({ ack: true, studyMode: isStudyModeOn });
    return true;
  }

  if (message.type === "GET_PAGE_SESSION_STATE") {
    let coveragePct = 0.0;
    if (currentSession) {
      if (currentSession.mediaType === "video") {
        coveragePct = calculateVideoCoverage(
          currentSession.watchedBuckets.size,
          currentSession.videoDurationSec || 1
        );
      } else {
        coveragePct = calculateArticleCoverage(
          currentSession.viewedWords,
          activeDwellSeconds,
          currentSession.totalWords
        );
      }
    }

    sendResponse({
      hasSession: !!currentSession,
      isTracking: isStudyModeOn && !!currentSession,
      pageTitle: currentSession?.pageTitle || document.title,
      sourceUrl: currentSession?.sourceUrl || window.location.href,
      sourceDomain: currentSession?.sourceDomain || window.location.hostname,
      mediaType: currentSession?.mediaType || "article",
      activeDwellSeconds,
      estimatedCoveragePct: coveragePct,
      topKeywords: currentSession?.topKeywords?.slice(0, 4) || [],
      rewindCount: currentSession?.rewindCount || 0,
    });
    return true;
  }

  if (message.type === "REQUEST_SESSION_FLUSH") {
    if (!currentSession) {
      sendResponse({ flushed: false, message: "No active session on this tab." });
      return true;
    }
    if (activeDwellSeconds < 15) {
      sendResponse({
        flushed: false,
        dwell: activeDwellSeconds,
        message: `Need at least 15s of active study time before syncing (current: ${activeDwellSeconds}s).`,
      });
      return true;
    }
    const eventPayload = buildSessionPayload("manual_sync_now");
    activeDwellSeconds = 0;
    currentSession.clientEventId = generateUUID();
    currentSession.startedAt = new Date().toISOString();
    sendResponse({
      flushed: true,
      event: eventPayload,
      dwell: eventPayload ? eventPayload.active_dwell_seconds : 0,
    });
    return true;
  }

  return false;
});

// -------------------------------------------------------------
// STORAGE LISTENER
// -------------------------------------------------------------
try {
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === "local" && changes.studyMode) {
      const wasStudyMode = isStudyModeOn;
      isStudyModeOn = changes.studyMode.newValue === true;
      console.info("[Cognitwin] Storage studyMode changed:", isStudyModeOn);
      if (isStudyModeOn && !wasStudyMode) {
        if (!currentSession) {
          initContentTracker(true);
        }
      } else if (!isStudyModeOn && wasStudyMode) {
        flushSession("study_mode_off", true);
      }
    }
  });
} catch {
  // Graceful fallback if storage listener unavailable
}

// -------------------------------------------------------------
// TRACKING INITIALIZATION
// -------------------------------------------------------------
async function initContentTracker(immediate: boolean = false): Promise<void> {
  const currentUrl = window.location.href;
  const urlInfo = sanitizeUrl(currentUrl);

  if (urlInfo.isIgnored) {
    console.info("[Cognitwin] URL ignored for tracking:", currentUrl);
    return;
  }

  try {
    const storage = await chrome.storage.local.get(["studyMode", "blocklist"]);
    isStudyModeOn = storage.studyMode === true;
    const blocklist: string[] = storage.blocklist || [];

    if (isBlockedDomain(urlInfo.domain, blocklist)) {
      console.info("[Cognitwin] Domain is on blocklist:", urlInfo.domain);
      return;
    }
  } catch (err) {
    console.warn("[Cognitwin] Storage access error:", err);
    return;
  }

  console.info("[Cognitwin] Initializing tracking session. Study Mode:", isStudyModeOn);
  if (immediate || document.readyState === "complete") {
    startTrackingSession(urlInfo);
  } else {
    setTimeout(() => {
      startTrackingSession(urlInfo);
    }, 1200);
  }
}

function startTrackingSession(urlInfo: ReturnType<typeof sanitizeUrl>): void {
  const now = new Date().toISOString();
  activeDwellSeconds = 0;
  lastHeartbeatTime = Date.now();

  if (urlInfo.mediaType === "video") {
    setupVideoObserver(urlInfo, now);
  } else if (urlInfo.mediaType === "pdf") {
    const { title, keywords } = extractKeywordsFromFilename(window.location.href);
    currentSession = {
      clientEventId: generateUUID(),
      sourceUrl: urlInfo.sanitizedUrl,
      sourceDomain: urlInfo.domain,
      pageTitle: title,
      mediaType: "pdf",
      startedAt: now,
      totalWords: 500,
      viewedWords: 500,
      viewedHeadings: [title],
      topKeywords: keywords,
      watchedBuckets: new Set(),
      rewindCount: 0,
      pauseCount: 0,
      lastVideoTime: 0,
      playbackRates: [],
    };
  } else {
    setupArticleObserver(urlInfo, now);
  }
}

function setupArticleObserver(urlInfo: ReturnType<typeof sanitizeUrl>, startTimeIso: string): void {
  const root =
    document.querySelector("main") ||
    document.querySelector("article") ||
    document.querySelector('[role="main"]') ||
    document.body;

  if (!root) return;

  // Clone and strip headers, footers, navs, aside
  const clone = root.cloneNode(true) as HTMLElement;
  const ignoreSelectors = ["header", "footer", "nav", "aside", ".nav", ".menu", ".sidebar", "script", "style"];
  ignoreSelectors.forEach((sel) => {
    clone.querySelectorAll(sel).forEach((el) => el.remove());
  });

  const rawText = clone.textContent || clone.innerText || "";
  const totalWords = Math.max(1, rawText.trim().split(/\s+/).length);

  // Extract headings
  const headingElements = Array.from(root.querySelectorAll("h1, h2, h3"));
  const rawHeadings = headingElements.map((h) => (h as HTMLElement).innerText || "");
  const headings = extractHeadings(rawHeadings, 5);

  // Extract keywords locally (text discarded immediately)
  const keywords = extractKeywords(rawText, 10);

  currentSession = {
    clientEventId: generateUUID(),
    sourceUrl: urlInfo.sanitizedUrl,
    sourceDomain: urlInfo.domain,
    pageTitle: document.title || urlInfo.domain,
    mediaType: urlInfo.mediaType,
    startedAt: startTimeIso,
    totalWords,
    viewedWords: Math.min(totalWords, 100),
    viewedHeadings: headings,
    topKeywords: keywords,
    watchedBuckets: new Set(),
    rewindCount: 0,
    pauseCount: 0,
    lastVideoTime: 0,
    playbackRates: [],
  };

  const handleScroll = () => {
    if (!currentSession) return;
    const scrollY = window.scrollY || window.pageYOffset;
    const viewportHeight = window.innerHeight;
    const documentHeight = Math.max(
      document.body.scrollHeight,
      document.documentElement.scrollHeight
    );
    const scrolledRatio = Math.min(1.0, (scrollY + viewportHeight) / (documentHeight || 1));
    const estimatedViewed = Math.round(scrolledRatio * currentSession.totalWords);
    currentSession.viewedWords = Math.max(currentSession.viewedWords, estimatedViewed);
  };

  window.addEventListener("scroll", handleScroll, { passive: true });
}

function setupVideoObserver(urlInfo: ReturnType<typeof sanitizeUrl>, startTimeIso: string): void {
  const video = document.querySelector("video");
  if (!video) {
    setTimeout(() => {
      const retryVideo = document.querySelector("video");
      if (retryVideo) attachVideoListeners(retryVideo, urlInfo, startTimeIso);
    }, 1500);
    return;
  }

  attachVideoListeners(video, urlInfo, startTimeIso);
}

function attachVideoListeners(
  video: HTMLVideoElement,
  urlInfo: ReturnType<typeof sanitizeUrl>,
  startTimeIso: string
): void {
  let channelName = "";
  const channelEl = document.querySelector("#channel-name, .ytd-channel-name, [itemprop='author']");
  if (channelEl) {
    channelName = ((channelEl as HTMLElement).innerText || "").trim();
  }

  const chapterTitles: string[] = [];
  document.querySelectorAll(".ytp-chapter-title-content, .ytd-macro-markers-list-item-renderer").forEach((el) => {
    const text = ((el as HTMLElement).innerText || "").trim();
    if (text) chapterTitles.push(text);
  });

  const session: ContentSession = {
    clientEventId: generateUUID(),
    sourceUrl: urlInfo.sanitizedUrl,
    sourceDomain: urlInfo.domain,
    pageTitle: document.title.replace(" - YouTube", "") || "Video",
    mediaType: "video",
    startedAt: startTimeIso,
    totalWords: 0,
    viewedWords: 0,
    viewedHeadings: extractHeadings(chapterTitles, 5),
    topKeywords: extractKeywords(document.title, 8),
    videoDurationSec: Math.round(video.duration || 0),
    channelName: channelName || undefined,
    watchedBuckets: new Set<number>(),
    rewindCount: 0,
    pauseCount: 0,
    lastVideoTime: video.currentTime,
    playbackRates: [video.playbackRate || 1.0],
  };

  currentSession = session;

  video.addEventListener("timeupdate", () => {
    if (!currentSession) return;
    const isAdShowing = !!document.querySelector(".ad-showing, .ad-interrupting");
    if (isAdShowing) return;

    const currentTime = video.currentTime;
    if (isRewindSeek(currentSession.lastVideoTime, currentTime)) {
      currentSession.rewindCount += 1;
    }
    currentSession.lastVideoTime = currentTime;

    const bucket = getBucketIndex(currentTime);
    currentSession.watchedBuckets.add(bucket);

    if (video.duration && !currentSession.videoDurationSec) {
      currentSession.videoDurationSec = Math.round(video.duration);
    }
  });

  video.addEventListener("ratechange", () => {
    if (currentSession && video.playbackRate) {
      currentSession.playbackRates.push(video.playbackRate);
    }
  });

  video.addEventListener("pause", () => {
    if (currentSession && !video.ended) {
      currentSession.pauseCount += 1;
    }
  });
}

// -------------------------------------------------------------
// DWELL ACCUMULATOR TICK
// -------------------------------------------------------------
function tick(): void {
  const now = Date.now();
  const deltaSec = Math.round((now - lastHeartbeatTime) / 1000);
  lastHeartbeatTime = now;

  let isActive = isTabVisible;

  if (currentSession?.mediaType === "video") {
    const video = document.querySelector("video");
    const isPlaying = video && !video.paused && !video.ended;
    isActive = isTabVisible && !!isPlaying;
  }

  if (isActive && isStudyModeOn && deltaSec > 0 && deltaSec <= 3) {
    activeDwellSeconds += deltaSec;
  }
}

setInterval(tick, 1000);

// -------------------------------------------------------------
// PAYLOAD BUILDER & FLUSH
// -------------------------------------------------------------
function buildSessionPayload(_reason: string) {
  if (!currentSession) return null;

  const dwellToSend = Math.max(15, activeDwellSeconds);

  let coveragePct = 0.0;
  if (currentSession.mediaType === "video") {
    coveragePct = calculateVideoCoverage(
      currentSession.watchedBuckets.size,
      currentSession.videoDurationSec || 1
    );
  } else {
    coveragePct = calculateArticleCoverage(
      currentSession.viewedWords,
      dwellToSend,
      currentSession.totalWords
    );
  }

  const avgPlaybackRate =
    currentSession.playbackRates.length > 0
      ? currentSession.playbackRates.reduce((a, b) => a + b, 0) / currentSession.playbackRates.length
      : 1.0;

  return {
    client_event_id: currentSession.clientEventId,
    source_domain: currentSession.sourceDomain,
    source_url: currentSession.sourceUrl,
    page_title: currentSession.pageTitle,
    media_type: currentSession.mediaType,
    active_dwell_seconds: dwellToSend,
    estimated_coverage_pct: coveragePct,
    viewed_headings: currentSession.viewedHeadings,
    top_keywords: currentSession.topKeywords,
    started_at: currentSession.startedAt,
    ended_at: new Date().toISOString(),
    video_metrics:
      currentSession.mediaType === "video"
        ? {
            channel_name: currentSession.channelName,
            video_duration_sec: currentSession.videoDurationSec,
            playback_rate_avg: Math.round(avgPlaybackRate * 100) / 100,
            rewind_count: currentSession.rewindCount,
            pause_count: currentSession.pauseCount,
          }
        : null,
  };
}

function flushSession(reason: string, force: boolean = false): boolean {
  if (!currentSession) {
    return false;
  }

  if (!isStudyModeOn && reason !== "study_mode_off") {
    return false;
  }

  if (activeDwellSeconds < 15 && !force) {
    return false;
  }

  if (activeDwellSeconds === 0) {
    return false;
  }

  const eventPayload = buildSessionPayload(reason);
  if (!eventPayload) return false;

  try {
    chrome.runtime.sendMessage({
      type: "FLUSH_TELEMETRY_EVENT",
      reason,
      event: eventPayload,
    });
    activeDwellSeconds = 0;
    currentSession.clientEventId = generateUUID();
    currentSession.startedAt = new Date().toISOString();
    return true;
  } catch {
    return false;
  }
}

// 30-second local checkpoint
setInterval(() => {
  flushSession("30s_checkpoint");
}, 30000);

document.addEventListener("visibilitychange", () => {
  isTabVisible = document.visibilityState === "visible";
  if (!isTabVisible) {
    flushSession("tab_hidden");
  }
});

window.addEventListener("beforeunload", () => {
  flushSession("page_leave");
});

window.addEventListener("yt-navigate-finish", () => {
  flushSession("yt_navigate_finish");
  currentSession = null;
  initContentTracker(true);
});

// Run tracker
initContentTracker();
