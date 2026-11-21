import { sanitizeUrl } from "../utils/url";
import { isBlockedDomain } from "../utils/blocklist";
import { calculateArticleCoverage } from "../utils/coverage";
import { extractKeywords, extractHeadings, extractKeywordsFromFilename } from "../utils/nlp";
import { calculateVideoCoverage, isRewindSeek, getBucketIndex } from "../utils/video";
let currentSession = null;
let activeDwellSeconds = 0;
let lastHeartbeatTime = Date.now();
let isWindowFocused = document.hasFocus();
let isTabVisible = document.visibilityState === "visible";
let isStudyModeOn = false;
// Generate UUIDv4 client side
function generateUUID() {
    if (typeof crypto !== "undefined" && crypto.randomUUID) {
        return crypto.randomUUID();
    }
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
        const r = (Math.random() * 16) | 0;
        const v = c === "x" ? r : (r & 0x3) | 0x8;
        return v.toString(16);
    });
}
/**
 * Initializes content tracking on the current page.
 */
async function initContentTracker() {
    const currentUrl = window.location.href;
    const urlInfo = sanitizeUrl(currentUrl);
    if (urlInfo.isIgnored)
        return;
    // Check storage for Study Mode and custom blocklist
    try {
        const storage = await chrome.storage.local.get(["studyMode", "blocklist"]);
        isStudyModeOn = storage.studyMode === true;
        const blocklist = storage.blocklist || [];
        if (isBlockedDomain(urlInfo.domain, blocklist)) {
            return;
        }
    }
    catch {
        // If extension context invalidated, exit gracefully
        return;
    }
    // Hydration delay (~2s) for SPA / dynamic content
    setTimeout(() => {
        startTrackingSession(urlInfo);
    }, 2000);
}
function startTrackingSession(urlInfo) {
    const now = new Date().toISOString();
    activeDwellSeconds = 0;
    lastHeartbeatTime = Date.now();
    if (urlInfo.mediaType === "video") {
        setupVideoObserver(urlInfo, now);
    }
    else if (urlInfo.mediaType === "pdf") {
        const { title, keywords } = extractKeywordsFromFilename(window.location.href);
        currentSession = {
            clientEventId: generateUUID(),
            sourceUrl: urlInfo.sanitizedUrl,
            sourceDomain: urlInfo.domain,
            pageTitle: title,
            mediaType: "pdf",
            startedAt: now,
            totalWords: 500, // Nominal estimation for PDF docs
            viewedWords: 500,
            viewedHeadings: [title],
            topKeywords: keywords,
            watchedBuckets: new Set(),
            rewindCount: 0,
            pauseCount: 0,
            lastVideoTime: 0,
            playbackRates: [],
        };
    }
    else {
        setupArticleObserver(urlInfo, now);
    }
}
/**
 * Extracts content from <main>, <article>, or fallback body minus navigation.
 */
function setupArticleObserver(urlInfo, startTimeIso) {
    const root = document.querySelector("main") ||
        document.querySelector("article") ||
        document.querySelector('[role="main"]') ||
        document.body;
    if (!root)
        return;
    // Clone and strip headers, footers, navs, aside
    const clone = root.cloneNode(true);
    const ignoreSelectors = ["header", "footer", "nav", "aside", ".nav", ".menu", ".sidebar", "script", "style"];
    ignoreSelectors.forEach((sel) => {
        clone.querySelectorAll(sel).forEach((el) => el.remove());
    });
    const rawText = clone.innerText || "";
    const totalWords = Math.max(1, rawText.trim().split(/\s+/).length);
    // Extract headings
    const headingElements = Array.from(root.querySelectorAll("h1, h2, h3"));
    const rawHeadings = headingElements.map((h) => h.innerText || "");
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
        viewedWords: Math.min(totalWords, 100), // initial viewport words
        viewedHeadings: headings,
        topKeywords: keywords,
        watchedBuckets: new Set(),
        rewindCount: 0,
        pauseCount: 0,
        lastVideoTime: 0,
        playbackRates: [],
    };
    // Scroll depth tracking
    const handleScroll = () => {
        if (!currentSession)
            return;
        const scrollY = window.scrollY || window.pageYOffset;
        const viewportHeight = window.innerHeight;
        const documentHeight = Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);
        const scrolledRatio = Math.min(1.0, (scrollY + viewportHeight) / (documentHeight || 1));
        const estimatedViewed = Math.round(scrolledRatio * currentSession.totalWords);
        currentSession.viewedWords = Math.max(currentSession.viewedWords, estimatedViewed);
    };
    window.addEventListener("scroll", handleScroll, { passive: true });
}
/**
 * Observes HTML5 / YouTube video player events.
 */
function setupVideoObserver(urlInfo, startTimeIso) {
    const video = document.querySelector("video");
    if (!video) {
        // Retry once if video element is still hydrating
        setTimeout(() => {
            const retryVideo = document.querySelector("video");
            if (retryVideo)
                attachVideoListeners(retryVideo, urlInfo, startTimeIso);
        }, 1500);
        return;
    }
    attachVideoListeners(video, urlInfo, startTimeIso);
}
function attachVideoListeners(video, urlInfo, startTimeIso) {
    // Extract channel name from YouTube DOM if available
    let channelName = "";
    const channelEl = document.querySelector("#channel-name, .ytd-channel-name, [itemprop='author']");
    if (channelEl) {
        channelName = (channelEl.innerText || "").trim();
    }
    // Extract chapters if present
    const chapterTitles = [];
    document.querySelectorAll(".ytp-chapter-title-content, .ytd-macro-markers-list-item-renderer").forEach((el) => {
        const text = (el.innerText || "").trim();
        if (text)
            chapterTitles.push(text);
    });
    const session = {
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
        watchedBuckets: new Set(),
        rewindCount: 0,
        pauseCount: 0,
        lastVideoTime: video.currentTime,
        playbackRates: [video.playbackRate || 1.0],
    };
    currentSession = session;
    video.addEventListener("timeupdate", () => {
        if (!currentSession)
            return;
        // Pause tracking when YouTube advertisement is showing
        const isAdShowing = !!document.querySelector(".ad-showing, .ad-interrupting");
        if (isAdShowing)
            return;
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
/**
 * Periodically increments dwell time and flushes checkpoints.
 */
function tick() {
    const now = Date.now();
    const deltaSec = Math.round((now - lastHeartbeatTime) / 1000);
    lastHeartbeatTime = now;
    // Attention rules:
    // For articles: requires active tab + focused window
    // For videos: counts if tab is visible and video playing
    let isActive = isWindowFocused && isTabVisible;
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
/**
 * Flushes the current session to the background worker.
 */
function flushSession(reason) {
    if (!currentSession || activeDwellSeconds < 15 || !isStudyModeOn) {
        return;
    }
    let coveragePct = 0.0;
    if (currentSession.mediaType === "video") {
        coveragePct = calculateVideoCoverage(currentSession.watchedBuckets.size, currentSession.videoDurationSec || 1);
    }
    else {
        coveragePct = calculateArticleCoverage(currentSession.viewedWords, activeDwellSeconds, currentSession.totalWords);
    }
    const avgPlaybackRate = currentSession.playbackRates.length > 0
        ? currentSession.playbackRates.reduce((a, b) => a + b, 0) / currentSession.playbackRates.length
        : 1.0;
    const eventPayload = {
        client_event_id: currentSession.clientEventId,
        source_domain: currentSession.sourceDomain,
        source_url: currentSession.sourceUrl,
        page_title: currentSession.pageTitle,
        media_type: currentSession.mediaType,
        active_dwell_seconds: activeDwellSeconds,
        estimated_coverage_pct: coveragePct,
        viewed_headings: currentSession.viewedHeadings,
        top_keywords: currentSession.topKeywords,
        started_at: currentSession.startedAt,
        ended_at: new Date().toISOString(),
        video_metrics: currentSession.mediaType === "video"
            ? {
                channel_name: currentSession.channelName,
                video_duration_sec: currentSession.videoDurationSec,
                playback_rate_avg: Math.round(avgPlaybackRate * 100) / 100,
                rewind_count: currentSession.rewindCount,
                pause_count: currentSession.pauseCount,
            }
            : null,
    };
    try {
        chrome.runtime.sendMessage({
            type: "FLUSH_TELEMETRY_EVENT",
            reason,
            event: eventPayload,
        });
    }
    catch {
        // Ignore runtime disconnected errors
    }
}
// 30-second local checkpoint
setInterval(() => {
    flushSession("30s_checkpoint");
}, 30000);
// Window and tab visibility listeners
window.addEventListener("focus", () => {
    isWindowFocused = true;
});
window.addEventListener("blur", () => {
    isWindowFocused = false;
});
document.addEventListener("visibilitychange", () => {
    isTabVisible = document.visibilityState === "visible";
    if (!isTabVisible) {
        flushSession("tab_hidden");
    }
});
window.addEventListener("beforeunload", () => {
    flushSession("page_leave");
});
// YouTube SPA transition observer
window.addEventListener("yt-navigate-finish", () => {
    flushSession("yt_navigate_finish");
    currentSession = null;
    initContentTracker();
});
// Listen for messages from background script
chrome.runtime.onMessage.addListener((message) => {
    if (message.type === "STUDY_MODE_CHANGED") {
        isStudyModeOn = message.studyMode === true;
        if (!isStudyModeOn) {
            flushSession("study_mode_off");
        }
    }
});
// Run tracker
initContentTracker();
