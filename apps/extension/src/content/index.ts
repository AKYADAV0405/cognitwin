console.info("[Cognitwin] content script active on", location.hostname, location.href);

import { sanitizeUrl } from "../utils/url";
import { isBlockedDomain } from "../utils/blocklist";
import { calculateArticleCoverage, hasMinimumViewportDwell } from "../utils/coverage";
import {
  extractKeywords,
  extractHeadings,
  extractKeywordsFromFilename,
  cleanPageTitle,
} from "../utils/nlp";
import {
  calculateVideoCoverage,
  isRewindSeek,
  getBucketIndex,
  shouldTrackVideoDwell,
  appendWatchedChapter,
} from "../utils/video";
import {
  WarmSessionState,
  calculateUnsyncedDwell,
  shouldFlushDelta,
  isWarmSessionValid,
} from "../utils/session";

interface ContentSession {
  clientEventId: string;
  sourceUrl: string;
  sourceDomain: string;
  pageTitle: string;
  metaDescription?: string | null;
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
let syncedDwellSeconds = 0;
let lastHeartbeatTime = Date.now();
let isTabVisible = document.visibilityState === "visible";
let isStudyModeOn = false;

// Viewport IntersectionObserver state for genuine reading
let articleObserver: IntersectionObserver | null = null;
const observedElementsMap = new Map<Element, number>(); // element -> enter timestamp ms
const countedParagraphs = new WeakSet<Element>();
const viewedHeadingsSet = new Set<string>();

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

function getSessionStorageKey(url: string): string {
  return `cognitwin_warm_${url}`;
}

async function getWarmSession(url: string): Promise<WarmSessionState | null> {
  try {
    if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.session) {
      const key = getSessionStorageKey(url);
      const res = await chrome.storage.session.get([key]);
      const session = res[key] as WarmSessionState | undefined;
      if (session && isWarmSessionValid(session, url)) {
        return session;
      }
    }
  } catch {
    // Session storage unavailable
  }
  return null;
}

async function persistWarmSession(): Promise<void> {
  if (!currentSession) return;
  try {
    if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.session) {
      const key = getSessionStorageKey(currentSession.sourceUrl);
      const warm: WarmSessionState = {
        sanitizedUrl: currentSession.sourceUrl,
        sourceDomain: currentSession.sourceDomain,
        pageTitle: currentSession.pageTitle,
        metaDescription: currentSession.metaDescription,
        mediaType: currentSession.mediaType,
        startedAt: currentSession.startedAt,
        activeDwellSeconds,
        syncedDwellSeconds,
        viewedWords: currentSession.viewedWords,
        totalWords: currentSession.totalWords,
        viewedHeadings: currentSession.viewedHeadings,
        topKeywords: currentSession.topKeywords,
        watchedBuckets: Array.from(currentSession.watchedBuckets),
        rewindCount: currentSession.rewindCount,
        pauseCount: currentSession.pauseCount,
        clientEventId: currentSession.clientEventId,
        lastUpdatedMs: Date.now(),
      };
      await chrome.storage.session.set({ [key]: warm });
    }
  } catch {
    // Ignore storage write errors
  }
}

function extractMetaDescription(): string | null {
  try {
    const meta = document.querySelector(
      'meta[name="description"], meta[property="og:description"], meta[name="twitter:description"]'
    ) as HTMLMetaElement | null;
    if (meta && meta.content) {
      const clean = meta.content.replace(/\s+/g, " ").trim();
      if (clean) return clean.slice(0, 200);
    }
  } catch {
    // Ignore DOM query errors
  }
  return null;
}

function extractYouTubeDescription(channelName: string, chapterTitles: string[]): string | null {
  try {
    const meta = document.querySelector('meta[name="description"]') as HTMLMetaElement | null;
    if (meta && meta.content) {
      const clean = meta.content.replace(/\s+/g, " ").trim();
      if (clean) return clean.slice(0, 200);
    }
    const descEl = document.querySelector(
      "#description-inline-expander, #description, ytd-text-inline-expander"
    ) as HTMLElement | null;
    if (descEl && descEl.textContent) {
      const clean = descEl.textContent.replace(/\s+/g, " ").trim();
      if (clean) return clean.slice(0, 200);
    }
  } catch {
    // Ignore DOM errors
  }

  const parts: string[] = [];
  if (channelName) parts.push(`Channel: ${channelName}`);
  if (chapterTitles.length > 0) parts.push(`Topics: ${chapterTitles.slice(0, 4).join(", ")}`);
  return parts.length > 0 ? parts.join(" | ").slice(0, 200) : null;
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
          currentSession.totalWords,
          220
        );
      }
    }

    sendResponse({
      hasSession: !!currentSession,
      isTracking: isStudyModeOn && !!currentSession && isTabVisible,
      pageTitle: currentSession?.pageTitle || cleanPageTitle(document.title),
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
    const unsyncedDwell = calculateUnsyncedDwell(activeDwellSeconds, syncedDwellSeconds);
    if (unsyncedDwell < 15) {
      const msg =
        syncedDwellSeconds > 0
          ? `Need at least 15s of new active study time before syncing (already synced: ${syncedDwellSeconds}s, current total: ${activeDwellSeconds}s).`
          : `Need at least 15s of active study time before syncing (current: ${activeDwellSeconds}s).`;
      sendResponse({
        flushed: false,
        dwell: activeDwellSeconds,
        message: msg,
      });
      return true;
    }

    const eventPayload = buildSessionPayload("manual_sync_now");
    syncedDwellSeconds = activeDwellSeconds;
    currentSession.clientEventId = generateUUID();
    persistWarmSession();

    sendResponse({
      flushed: true,
      event: eventPayload,
      dwell: activeDwellSeconds,
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
async function initContentTracker(immediate: boolean = false, overrideTitle?: string): Promise<void> {
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
    startTrackingSession(urlInfo, overrideTitle);
  } else {
    setTimeout(() => {
      startTrackingSession(urlInfo, overrideTitle);
    }, 1200);
  }
}

async function startTrackingSession(
  urlInfo: ReturnType<typeof sanitizeUrl>,
  overrideTitle?: string
): Promise<void> {
  const now = new Date().toISOString();
  lastHeartbeatTime = Date.now();

  // Check for cached warm session on same URL
  const warm = await getWarmSession(urlInfo.sanitizedUrl);
  if (warm) {
    console.info("[Cognitwin] Resuming warm session for:", urlInfo.sanitizedUrl, "dwell:", warm.activeDwellSeconds);
    activeDwellSeconds = warm.activeDwellSeconds;
    syncedDwellSeconds = warm.syncedDwellSeconds;
  } else {
    activeDwellSeconds = 0;
    syncedDwellSeconds = 0;
  }

  if (urlInfo.mediaType === "video") {
    setupVideoObserver(urlInfo, now, warm, overrideTitle);
  } else if (urlInfo.mediaType === "pdf") {
    const { title, keywords } = extractKeywordsFromFilename(window.location.href);
    const finalTitle = overrideTitle || warm?.pageTitle || title;
    currentSession = {
      clientEventId: warm?.clientEventId || generateUUID(),
      sourceUrl: urlInfo.sanitizedUrl,
      sourceDomain: urlInfo.domain,
      pageTitle: finalTitle,
      metaDescription: `PDF Document: ${finalTitle}`,
      mediaType: "pdf",
      startedAt: warm?.startedAt || now,
      totalWords: 2500,
      viewedWords: warm?.viewedWords || 250,
      viewedHeadings: warm?.viewedHeadings || [finalTitle],
      topKeywords: warm?.topKeywords || keywords,
      watchedBuckets: new Set(),
      rewindCount: 0,
      pauseCount: 0,
      lastVideoTime: 0,
      playbackRates: [],
    };
  } else {
    setupArticleObserver(urlInfo, now, warm, overrideTitle);
  }
}

function setupArticleObserver(
  urlInfo: ReturnType<typeof sanitizeUrl>,
  startTimeIso: string,
  warm?: WarmSessionState | null,
  overrideTitle?: string
): void {
  if (articleObserver) {
    articleObserver.disconnect();
    observedElementsMap.clear();
  }

  const root =
    document.querySelector("main") ||
    document.querySelector("article") ||
    document.querySelector('[role="main"]') ||
    document.body;

  if (!root) return;

  const cleanTitle = overrideTitle || cleanPageTitle(document.title || urlInfo.domain);
  const metaDescription = extractMetaDescription();

  // Clone and strip headers, footers, navs, aside
  const clone = root.cloneNode(true) as HTMLElement;
  const ignoreSelectors = ["header", "footer", "nav", "aside", ".nav", ".menu", ".sidebar", "script", "style"];
  ignoreSelectors.forEach((sel) => {
    clone.querySelectorAll(sel).forEach((el) => el.remove());
  });

  const rawText = clone.textContent || clone.innerText || "";
  const totalWords = Math.max(1, rawText.trim().split(/\s+/).filter(Boolean).length);

  // Initialize viewed headings set
  viewedHeadingsSet.clear();
  const initialHeadings = warm?.viewedHeadings || [];
  initialHeadings.forEach((h) => viewedHeadingsSet.add(h));

  // Extract code snippets from <code> and <pre> tags for 3x technical weighting
  const codeElements = Array.from(root.querySelectorAll("code, pre"));
  const codeSnippets = codeElements
    .map((el) => (el as HTMLElement).textContent || "")
    .filter((t) => t.trim().length > 0)
    .slice(0, 10);

  // High-signal keywords with 3x weighting for title/headings/code and 2-word technical bigrams
  const keywords = extractKeywords(rawText, {
    title: cleanTitle,
    headings: Array.from(viewedHeadingsSet),
    codeSnippets,
    maxKeywords: 10,
  });

  currentSession = {
    clientEventId: warm?.clientEventId || generateUUID(),
    sourceUrl: urlInfo.sanitizedUrl,
    sourceDomain: urlInfo.domain,
    pageTitle: cleanTitle,
    metaDescription,
    mediaType: urlInfo.mediaType,
    startedAt: warm?.startedAt || startTimeIso,
    totalWords,
    viewedWords: warm?.viewedWords || 0,
    viewedHeadings: Array.from(viewedHeadingsSet),
    topKeywords: keywords,
    watchedBuckets: new Set(),
    rewindCount: 0,
    pauseCount: 0,
    lastVideoTime: 0,
    playbackRates: [],
  };

  // IntersectionObserver requiring elements stay visible in viewport for >= 2 seconds
  if (typeof IntersectionObserver !== "undefined") {
    articleObserver = new IntersectionObserver(
      (entries) => {
        const now = Date.now();
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            observedElementsMap.set(entry.target, now);
          } else {
            const enteredAt = observedElementsMap.get(entry.target);
            if (enteredAt && hasMinimumViewportDwell(enteredAt, now, 2000)) {
              markElementViewed(entry.target);
            }
            observedElementsMap.delete(entry.target);
          }
        });
      },
      { threshold: 0.5 }
    );

    const targets = root.querySelectorAll("h1, h2, h3, p");
    targets.forEach((el) => articleObserver?.observe(el));
  }
}

function markElementViewed(el: Element): void {
  if (!currentSession) return;
  const tag = el.tagName.toUpperCase();

  if (tag === "P") {
    if (!countedParagraphs.has(el)) {
      countedParagraphs.add(el);
      const text = (el.textContent || "").trim();
      const count = text.split(/\s+/).filter(Boolean).length;
      if (count > 0) {
        currentSession.viewedWords = Math.min(
          currentSession.totalWords,
          currentSession.viewedWords + count
        );
      }
    }
  } else if (tag === "H1" || tag === "H2" || tag === "H3") {
    const text = (el.textContent || "").trim();
    if (text.length >= 3) {
      const lower = text.toLowerCase();
      let alreadyPresent = false;
      for (const existing of viewedHeadingsSet) {
        if (existing.toLowerCase() === lower) {
          alreadyPresent = true;
          break;
        }
      }
      if (!alreadyPresent && viewedHeadingsSet.size < 6) {
        viewedHeadingsSet.add(text);
        currentSession.viewedHeadings = extractHeadings(Array.from(viewedHeadingsSet), 6);
      }
    }
  }
}

function setupVideoObserver(
  urlInfo: ReturnType<typeof sanitizeUrl>,
  startTimeIso: string,
  warm?: WarmSessionState | null,
  overrideTitle?: string
): void {
  const video = document.querySelector("video");
  if (!video) {
    setTimeout(() => {
      const retryVideo = document.querySelector("video");
      if (retryVideo) attachVideoListeners(retryVideo, urlInfo, startTimeIso, warm, overrideTitle);
    }, 1500);
    return;
  }

  attachVideoListeners(video, urlInfo, startTimeIso, warm, overrideTitle);
}

function attachVideoListeners(
  video: HTMLVideoElement,
  urlInfo: ReturnType<typeof sanitizeUrl>,
  startTimeIso: string,
  warm?: WarmSessionState | null,
  overrideTitle?: string
): void {
  let channelName = "";
  const channelEl = document.querySelector("#channel-name, .ytd-channel-name, [itemprop='author']");
  if (channelEl) {
    channelName = ((channelEl as HTMLElement).innerText || "").trim();
  }

  const chapterTitles: string[] = warm?.viewedHeadings ? [...warm.viewedHeadings] : [];
  document.querySelectorAll(".ytp-chapter-title-content, .ytd-macro-markers-list-item-renderer").forEach((el) => {
    const text = ((el as HTMLElement).innerText || "").trim();
    if (text) chapterTitles.push(text);
  });

  const cleanTitle = overrideTitle || cleanPageTitle(document.title || "Video");
  const metaDescription = extractYouTubeDescription(channelName, chapterTitles);
  const headings = extractHeadings(chapterTitles, 6);
  const keywords = extractKeywords(cleanTitle, {
    title: cleanTitle,
    headings,
    maxKeywords: 10,
  });

  const session: ContentSession = {
    clientEventId: warm?.clientEventId || generateUUID(),
    sourceUrl: urlInfo.sanitizedUrl,
    sourceDomain: urlInfo.domain,
    pageTitle: cleanTitle,
    metaDescription,
    mediaType: "video",
    startedAt: warm?.startedAt || startTimeIso,
    totalWords: 0,
    viewedWords: 0,
    viewedHeadings: headings,
    topKeywords: keywords,
    videoDurationSec: Math.round(video.duration || 0),
    channelName: channelName || undefined,
    watchedBuckets: new Set<number>(warm?.watchedBuckets || []),
    rewindCount: warm?.rewindCount || 0,
    pauseCount: warm?.pauseCount || 0,
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

    // Capture watched chapters during active playback
    const currentChapterEl = document.querySelector(".ytp-chapter-title-content");
    if (currentChapterEl && currentChapterEl.textContent) {
      currentSession.viewedHeadings = appendWatchedChapter(
        currentSession.viewedHeadings,
        currentChapterEl.textContent,
        6
      );
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

  // Genuine Article Viewport Reading: check elements continuously visible for >= 2s
  if (articleObserver && observedElementsMap.size > 0) {
    observedElementsMap.forEach((enteredAt, el) => {
      if (hasMinimumViewportDwell(enteredAt, now, 2000)) {
        markElementViewed(el);
      }
    });
  }

  // Genuine YouTube Lecture Guardrail: Muted tab pauses UNLESS captions are visible
  if (currentSession?.mediaType === "video") {
    const video = document.querySelector("video");
    const isPlaying = !!(video && !video.paused && !video.ended);
    const isMuted = !!(video && (video.muted || video.volume === 0));

    // Check for active visible YouTube captions
    const captionEl = document.querySelector(".ytp-caption-window-container, .caption-window");
    const hasCaptions = !!(captionEl && (captionEl.textContent || "").trim().length > 0);
    const isAdShowing = !!document.querySelector(".ad-showing, .ad-interrupting");

    const canTrackVideo = shouldTrackVideoDwell(isPlaying, isMuted, hasCaptions) && !isAdShowing;
    isActive = isTabVisible && canTrackVideo;
  }

  if (isActive && isStudyModeOn && deltaSec > 0 && deltaSec <= 3) {
    activeDwellSeconds += deltaSec;
  }
}

setInterval(tick, 1000);

// -------------------------------------------------------------
// PAYLOAD BUILDER & INCREMENTAL FLUSH
// -------------------------------------------------------------
function buildSessionPayload(_reason: string) {
  if (!currentSession) return null;

  const unsyncedDwell = calculateUnsyncedDwell(activeDwellSeconds, syncedDwellSeconds);
  const dwellToSend = Math.max(15, unsyncedDwell);

  let coveragePct = 0.0;
  if (currentSession.mediaType === "video") {
    coveragePct = calculateVideoCoverage(
      currentSession.watchedBuckets.size,
      currentSession.videoDurationSec || 1
    );
  } else {
    coveragePct = calculateArticleCoverage(
      currentSession.viewedWords,
      activeDwellSeconds,
      currentSession.totalWords,
      220
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
    meta_description: currentSession.metaDescription || null,
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

  const unsyncedDwell = calculateUnsyncedDwell(activeDwellSeconds, syncedDwellSeconds);
  if (!shouldFlushDelta(unsyncedDwell, 15, force)) {
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
    syncedDwellSeconds = activeDwellSeconds;
    currentSession.clientEventId = generateUUID();
    persistWarmSession();
    return true;
  } catch {
    return false;
  }
}

// 30-second local checkpoint
setInterval(() => {
  flushSession("30s_checkpoint");
  persistWarmSession();
}, 30000);

// TAB SWITCH / VISIBILITY CHANGE: PAUSE & RESUME (Never reset active dwell or state)
document.addEventListener("visibilitychange", () => {
  isTabVisible = document.visibilityState === "visible";
  lastHeartbeatTime = Date.now();
  if (!isTabVisible) {
    // PAUSE: preserve activeDwellSeconds and persist warm session state
    persistWarmSession();
  } else {
    // RESUME: continue counting seamlessly
    lastHeartbeatTime = Date.now();
  }
});

window.addEventListener("beforeunload", () => {
  flushSession("page_leave");
  persistWarmSession();
});

// YouTube SPA Navigation with Title Debouncing
let ytNavTimeout: ReturnType<typeof setTimeout> | null = null;
window.addEventListener("yt-navigate-finish", () => {
  if (ytNavTimeout) clearTimeout(ytNavTimeout);
  flushSession("yt_navigate_finish");
  currentSession = null;

  // Debounce and poll for watch-page title DOM update (up to 1.5s)
  const previousTitle = cleanPageTitle(document.title || "");
  const startTime = Date.now();

  const checkTitle = () => {
    const metaTitleEl = document.querySelector(
      "ytd-watch-metadata h1 yt-formatted-string, #title h1 yt-formatted-string, h1.ytd-watch-metadata"
    );
    const candidate = metaTitleEl?.textContent?.trim() || "";
    const docTitle = cleanPageTitle(document.title || "");
    const titleToUse = candidate || docTitle;

    if (titleToUse && titleToUse !== previousTitle && !titleToUse.toLowerCase().includes("youtube")) {
      initContentTracker(true, cleanPageTitle(titleToUse));
      return;
    }

    if (Date.now() - startTime >= 1500) {
      initContentTracker(true, cleanPageTitle(document.title || "Video"));
      return;
    }

    ytNavTimeout = setTimeout(checkTitle, 150);
  };

  ytNavTimeout = setTimeout(checkTitle, 150);
});

// Run tracker
initContentTracker();
