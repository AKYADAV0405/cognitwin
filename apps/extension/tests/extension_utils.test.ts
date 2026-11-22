import { describe, it, expect } from "vitest";
import { sanitizeUrl, isPdfUrl } from "../src/utils/url";
import { isBlockedDomain } from "../src/utils/blocklist";
import { calculateArticleCoverage, hasMinimumViewportDwell } from "../src/utils/coverage";
import {
  extractKeywords,
  extractHeadings,
  extractKeywordsFromFilename,
  cleanPageTitle,
} from "../src/utils/nlp";
import { checkRevisit, updateRevisitCache } from "../src/utils/revisit";
import {
  calculateVideoCoverage,
  isRewindSeek,
  getBucketIndex,
  shouldTrackVideoDwell,
  extractChapterTitle,
  appendWatchedChapter,
} from "../src/utils/video";
import { processSleepGap, coalesceTelemetryEvents, RawTelemetryEvent } from "../src/utils/coalesce";
import {
  calculateUnsyncedDwell,
  shouldFlushDelta,
  isWarmSessionValid,
  WarmSessionState,
} from "../src/utils/session";
import { createPdfSession, calculatePdfCoverage } from "../src/utils/pdf";

describe("1. URL Sanitization & Title Cleaning", () => {
  it("strips fragments and query parameters from standard articles", () => {
    const raw = "https://example.com/blog/deep-learning-intro?utm_source=twitter&session_id=123#summary";
    const res = sanitizeUrl(raw);
    expect(res.isIgnored).toBe(false);
    expect(res.sanitizedUrl).toBe("https://example.com/blog/deep-learning-intro");
    expect(res.domain).toBe("example.com");
    expect(res.mediaType).toBe("article");
  });

  it("preserves 'v' query param on youtube.com/watch and strips others", () => {
    const yt = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s&list=PL12345&ab_channel=RickAstley";
    const res = sanitizeUrl(yt);
    expect(res.isIgnored).toBe(false);
    expect(res.sanitizedUrl).toBe("https://www.youtube.com/watch?v=dQw4w9WgXcQ");
    expect(res.domain).toBe("youtube.com");
    expect(res.mediaType).toBe("video");
  });

  it("ignores YouTube Shorts and browser internal schemes", () => {
    expect(sanitizeUrl("https://www.youtube.com/shorts/xyz123").isIgnored).toBe(true);
    expect(sanitizeUrl("chrome://extensions").isIgnored).toBe(true);
    expect(sanitizeUrl("about:blank").isIgnored).toBe(true);
    expect(sanitizeUrl("http://localhost:5173/dashboard").isIgnored).toBe(true);
  });

  it("classifies documentation and PDF media types correctly", () => {
    const docs = sanitizeUrl("https://docs.python.org/3/library/asyncio.html");
    expect(docs.mediaType).toBe("docs");

    const pdf = sanitizeUrl("https://arxiv.org/pdf/1706.03762.pdf");
    expect(pdf.mediaType).toBe("pdf");
  });

  it("cleans site branding suffixes and notification badges from page titles", () => {
    expect(cleanPageTitle("Data Preprocessing in Python - GeeksforGeeks")).toBe("Data Preprocessing in Python");
    expect(cleanPageTitle("Array.prototype.map() - JavaScript | MDN")).toBe("Array.prototype.map() - JavaScript");
    expect(cleanPageTitle("Distributed Systems Lecture 1 - YouTube")).toBe("Distributed Systems Lecture 1");
    expect(cleanPageTitle("(3) How to invert binary tree - Stack Overflow")).toBe("How to invert binary tree");
  });
});

describe("2. Blocklist Filtering", () => {
  it("blocks social media, webmail, and banking domains", () => {
    expect(isBlockedDomain("mail.google.com")).toBe(true);
    expect(isBlockedDomain("https://web.whatsapp.com/")).toBe(true);
    expect(isBlockedDomain("https://instagram.com/p/12345")).toBe(true);
    expect(isBlockedDomain("chase.com")).toBe(true);
    expect(isBlockedDomain("subdomain.wellsfargo.com")).toBe(true);
  });

  it("allows learning and technical documentation sites", () => {
    expect(isBlockedDomain("react.dev")).toBe(false);
    expect(isBlockedDomain("https://arxiv.org/abs/2301.00001")).toBe(false);
    expect(isBlockedDomain("https://youtube.com/watch?v=123")).toBe(false);
  });
});

describe("3. Reading-Speed Coverage Calculation", () => {
  it("calculates realistic coverage based on reading speed and scroll depth", () => {
    const cov = calculateArticleCoverage(500, 150, 1000, 200);
    expect(cov).toBe(0.5);
  });

  it("prevents fast-scrolling cheat: dwell time caps viewed words", () => {
    const cov = calculateArticleCoverage(1000, 15, 1000, 200);
    expect(cov).toBe(0.05);
  });

  it("handles edge cases safely without NaN", () => {
    expect(calculateArticleCoverage(0, 0, 0)).toBe(0.0);
    expect(calculateArticleCoverage(100, 100, -10)).toBe(0.0);
  });
});

describe("4. High-Signal Keyword Extraction, Bigrams & Headings", () => {
  it("extracts 2-word technical bigrams and weights title/headings/code 3x higher", () => {
    const body = `
      In this tutorial step, we examine data preprocessing techniques.
      This method provides an example code block for machine learning pipelines.
      The output shows how data preprocessing and feature scaling improve accuracy.
    `;
    const keywords = extractKeywords(body, {
      title: "Data Preprocessing in Python",
      headings: ["Machine Learning Overview", "Feature Scaling Steps"],
      codeSnippets: ["import sklearn.preprocessing", "from sklearn.pipeline import Pipeline"],
      maxKeywords: 8,
    });

    // Should include high-signal 2-word bigrams
    expect(keywords).toContain("data preprocessing");
    expect(keywords).toContain("machine learning");

    // Should filter generic example nouns
    expect(keywords).not.toContain("tutorial");
    expect(keywords).not.toContain("step");
    expect(keywords).not.toContain("example");
    expect(keywords).not.toContain("output");
  });

  it("sanitizes headings and caps at 6 items <= 100 chars", () => {
    const headings = [
      "  Introduction to Distributed Systems  ",
      "Very long heading ".repeat(10),
      "Introduction to Distributed Systems", // duplicate
      "CAP Theorem and Consistency",
      "Raft Consensus Protocol",
      "Paxos State Machine Replication",
      "Byzantine Fault Tolerance",
      "Seventh Extra Heading Exceeding Limit",
    ];
    const cleaned = extractHeadings(headings, 6);
    expect(cleaned.length).toBe(6);
    expect(cleaned[0]).toBe("Introduction to Distributed Systems");
    expect(cleaned[1].length).toBeLessThanOrEqual(100);
    expect(cleaned).not.toContain("Seventh Extra Heading Exceeding Limit");
  });

  it("extracts title and keywords from PDF filename without DOM scraping", () => {
    const { title, keywords } = extractKeywordsFromFilename(
      "https://example.com/papers/Attention_Is_All_You_Need_Transformer_Architecture.pdf"
    );
    expect(title).toBe("Attention Is All You Need Transformer Architecture");
    expect(keywords).toContain("attention");
    expect(keywords).toContain("transformer");
  });
});

describe("5. 6-Hour Revisit Threshold & 14-Day Cache", () => {
  const now = Date.now();
  const ONE_HOUR_MS = 60 * 60 * 1000;
  const ONE_DAY_MS = 24 * ONE_HOUR_MS;

  it("only marks is_revisit=true when previous visit was at least 6 hours earlier (>= 0.25 days)", () => {
    const cache = {
      // Visited 2 hours ago (< 6h -> not a decay revisit)
      "https://react.dev/learn": {
        last_visited_at: now - 2 * ONE_HOUR_MS,
        visit_count: 1,
        max_coverage_pct: 0.4,
      },
      // Visited 8 hours ago (>= 6h -> is_revisit = true)
      "https://docs.python.org": {
        last_visited_at: now - 8 * ONE_HOUR_MS,
        visit_count: 1,
        max_coverage_pct: 0.5,
      },
      // Visited 3 days ago -> is_revisit = true
      "https://geeksforgeeks.org/ml": {
        last_visited_at: now - 3 * ONE_DAY_MS,
        visit_count: 2,
        max_coverage_pct: 0.7,
      },
    };

    // 2 hours ago (< 6h)
    const recent = checkRevisit(cache, "https://react.dev/learn", now);
    expect(recent.isRevisit).toBe(false);
    expect(recent.daysSinceLastVisit).toBe(0.08); // 2/24

    // 8 hours ago (>= 6h)
    const eightHours = checkRevisit(cache, "https://docs.python.org", now);
    expect(eightHours.isRevisit).toBe(true);
    expect(eightHours.daysSinceLastVisit).toBe(0.33); // 8/24 >= 0.25

    // 3 days ago
    const threeDays = checkRevisit(cache, "https://geeksforgeeks.org/ml", now);
    expect(threeDays.isRevisit).toBe(true);
    expect(threeDays.daysSinceLastVisit).toBe(3);

    // Non-existent
    const nonExistent = checkRevisit(cache, "https://unknown.com", now);
    expect(nonExistent.isRevisit).toBe(false);
    expect(nonExistent.daysSinceLastVisit).toBeNull();
  });

  it("auto-prunes entries older than 14 days on update", () => {
    const oldCache = {
      "https://old-article.com": {
        last_visited_at: now - 16 * ONE_DAY_MS,
        visit_count: 1,
        max_coverage_pct: 0.8,
      },
      "https://recent-article.com": {
        last_visited_at: now - 2 * ONE_DAY_MS,
        visit_count: 1,
        max_coverage_pct: 0.4,
      },
    };

    const updated = updateRevisitCache(oldCache, "https://new-article.com", 0.5, now);
    expect(updated["https://old-article.com"]).toBeUndefined();
    expect(updated["https://recent-article.com"]).toBeDefined();
    expect(updated["https://new-article.com"]).toBeDefined();
  });
});

describe("6. YouTube Video Observer & Rewind Counting", () => {
  it("computes coverage from unique 10s timeline buckets", () => {
    const cov = calculateVideoCoverage(4, 100);
    expect(cov).toBe(0.4);
  });

  it("detects rewinds (backward seek > 5s)", () => {
    expect(isRewindSeek(45, 30)).toBe(true);
    expect(isRewindSeek(45, 43)).toBe(false);
    expect(isRewindSeek(45, 55)).toBe(false);
  });

  it("computes bucket indices accurately", () => {
    expect(getBucketIndex(0)).toBe(0);
    expect(getBucketIndex(9.9)).toBe(0);
    expect(getBucketIndex(10.0)).toBe(1);
    expect(getBucketIndex(25.4)).toBe(2);
  });
});

describe("7. Sleep Gap Delta Cap & Coalescing Engine", () => {
  it("discards laptop sleep gaps > 90s and caps single delta at 60s", () => {
    const t0 = 1000000;
    const normal = processSleepGap(t0, t0 + 30000);
    expect(normal.isSleepGap).toBe(false);
    expect(normal.effectiveDeltaSec).toBe(30);

    const capped = processSleepGap(t0, t0 + 75000);
    expect(capped.isSleepGap).toBe(false);
    expect(capped.effectiveDeltaSec).toBe(60);

    const sleep = processSleepGap(t0, t0 + 300000);
    expect(sleep.isSleepGap).toBe(true);
    expect(sleep.effectiveDeltaSec).toBe(0);
  });

  it("drops visits < 15s, merges meta_description, and coalesces up to 6 headings & 12 keywords", () => {
    const events: RawTelemetryEvent[] = [
      {
        client_event_id: "evt-1",
        source_domain: "react.dev",
        source_url: "https://react.dev/learn",
        page_title: "Quick Start",
        media_type: "docs",
        active_dwell_seconds: 10, // < 15s -> should be dropped
        estimated_coverage_pct: 0.1,
        viewed_headings: ["Intro"],
        top_keywords: ["react"],
        is_revisit: false,
        days_since_last_visit: null,
        started_at: "2026-11-21T10:05:00Z",
        ended_at: "2026-11-21T10:05:10Z",
      },
      {
        client_event_id: "evt-2",
        source_domain: "react.dev",
        source_url: "https://react.dev/learn",
        page_title: "Quick Start",
        meta_description: "Learn React concepts and hooks.",
        media_type: "docs",
        active_dwell_seconds: 45,
        estimated_coverage_pct: 0.3,
        viewed_headings: ["Intro", "Components"],
        top_keywords: ["react", "component"],
        is_revisit: false,
        days_since_last_visit: null,
        started_at: "2026-11-21T10:10:00Z",
        ended_at: "2026-11-21T10:10:45Z",
      },
      {
        client_event_id: "evt-3",
        source_domain: "react.dev",
        source_url: "https://react.dev/learn",
        page_title: "Quick Start",
        media_type: "docs",
        active_dwell_seconds: 60,
        estimated_coverage_pct: 0.7,
        viewed_headings: ["Hooks", "Managing State"],
        top_keywords: ["hooks", "state", "useeffect"],
        is_revisit: false,
        days_since_last_visit: null,
        started_at: "2026-11-21T10:20:00Z",
        ended_at: "2026-11-21T10:21:00Z",
      },
    ];

    const coalesced = coalesceTelemetryEvents(events);
    expect(coalesced.length).toBe(1);
    expect(coalesced[0].active_dwell_seconds).toBe(105); // 45 + 60
    expect(coalesced[0].estimated_coverage_pct).toBe(0.7);
    expect(coalesced[0].meta_description).toBe("Learn React concepts and hooks.");
    expect(coalesced[0].viewed_headings).toContain("Intro");
    expect(coalesced[0].viewed_headings).toContain("Hooks");
    expect(coalesced[0].top_keywords).toContain("state");
  });
});

describe("8. Tab-Switch Pause/Resume & Incremental Dwell (No Reset, No Double-Counting)", () => {
  it("computes unsynced dwell delta correctly without double-counting", () => {
    // Session starts: 20s active dwell, 0s synced
    expect(calculateUnsyncedDwell(20, 0)).toBe(20);
    expect(shouldFlushDelta(20, 15)).toBe(true);

    // After syncing first 20s: total=20s, synced=20s -> unsynced=0s
    expect(calculateUnsyncedDwell(20, 20)).toBe(0);
    expect(shouldFlushDelta(0, 15)).toBe(false);

    // User switches tab (pause) and returns (resume): studies 10s more -> total=30s
    expect(calculateUnsyncedDwell(30, 20)).toBe(10);
    expect(shouldFlushDelta(10, 15)).toBe(false); // below 15s delta threshold

    // Studies another 15s -> total=45s, synced=20s -> unsynced=25s
    expect(calculateUnsyncedDwell(45, 20)).toBe(25);
    expect(shouldFlushDelta(25, 15)).toBe(true);

    // After second sync: synced becomes 45s, unsynced becomes 0s
    expect(calculateUnsyncedDwell(45, 45)).toBe(0);
  });

  it("validates warm session within 1-hour window and rejects stale/mismatched URLs", () => {
    const now = Date.now();
    const warm: WarmSessionState = {
      sanitizedUrl: "https://geeksforgeeks.org/data-preprocessing",
      sourceDomain: "geeksforgeeks.org",
      pageTitle: "Data Preprocessing",
      mediaType: "article",
      startedAt: new Date(now - 300000).toISOString(),
      activeDwellSeconds: 45,
      syncedDwellSeconds: 20,
      viewedWords: 150,
      totalWords: 500,
      viewedHeadings: ["Introduction", "Scaling"],
      topKeywords: ["data", "preprocessing"],
      clientEventId: "warm-evt-1",
      lastUpdatedMs: now - 60000, // 1 minute ago
    };

    // Same URL and fresh (< 1h) -> valid resume
    expect(isWarmSessionValid(warm, "https://geeksforgeeks.org/data-preprocessing", 3600000, now)).toBe(true);

    // Different URL -> invalid
    expect(isWarmSessionValid(warm, "https://react.dev/learn", 3600000, now)).toBe(false);

    // Stale session (> 1h ago) -> invalid
    expect(isWarmSessionValid(warm, "https://geeksforgeeks.org/data-preprocessing", 3600000, now + 4000000)).toBe(false);

    // Null session -> invalid
    expect(isWarmSessionValid(null, "https://geeksforgeeks.org/data-preprocessing")).toBe(false);
  });
});

describe("9. Genuine Browser PDF Tracking (Background)", () => {
  it("detects PDF URLs (.pdf and /pdf/ paths like arXiv) accurately", () => {
    expect(isPdfUrl("https://arxiv.org/pdf/1706.03762.pdf")).toBe(true);
    expect(isPdfUrl("https://arxiv.org/pdf/1706.03762")).toBe(true);
    expect(isPdfUrl("https://example.com/docs/paper.PDF")).toBe(true);
    expect(isPdfUrl("https://example.com/viewer", "Attention Is All You Need.pdf")).toBe(true);
    expect(isPdfUrl("https://example.com/viewer", "Research Report [PDF]")).toBe(true);

    // Non-PDF URLs
    expect(isPdfUrl("https://react.dev/learn")).toBe(false);
    expect(isPdfUrl("https://youtube.com/watch?v=123")).toBe(false);
    expect(isPdfUrl("https://docs.python.org/3/tutorial/")).toBe(false);
  });

  it("creates and initializes a background PDF session", () => {
    const session = createPdfSession(
      101,
      "https://arxiv.org/pdf/1706.03762",
      "arxiv.org",
      "Attention Is All You Need.pdf"
    );

    expect(session.tabId).toBe(101);
    expect(session.mediaType).toBe("pdf" as const);
    expect(session.pageTitle).toBe("Attention Is All You Need");
    expect(session.sourceDomain).toBe("arxiv.org");
    expect(session.activeDwellSeconds).toBe(0);
    expect(session.syncedDwellSeconds).toBe(0);
    expect(session.topKeywords.length).toBeGreaterThan(0);
  });

  it("computes reading-speed bounded coverage for PDFs at 220 WPM", () => {
    // 60s dwell on 2500-word paper: 60 * (220/60) = 220 words / 2500 = 0.088
    const cov = calculatePdfCoverage(60, 2500, 220);
    expect(cov).toBe(0.088);

    // 0s dwell -> 0
    expect(calculatePdfCoverage(0, 2500, 220)).toBe(0.0);
  });
});

describe("10. YouTube Lecture Guardrails: Muted Without Captions & Chapter Capture", () => {
  it("pauses video dwell when muted UNLESS active captions are visible", () => {
    // Playing + Unmuted -> Track
    expect(shouldTrackVideoDwell(true, false, false)).toBe(true);

    // Playing + Muted + No Captions -> PAUSE (false)
    expect(shouldTrackVideoDwell(true, true, false)).toBe(false);

    // Playing + Muted + Active Captions Visible -> TRACK (true)
    expect(shouldTrackVideoDwell(true, true, true)).toBe(true);

    // Video Paused -> PAUSE (false) regardless of audio/captions
    expect(shouldTrackVideoDwell(false, false, false)).toBe(false);
    expect(shouldTrackVideoDwell(false, true, true)).toBe(false);
  });

  it("extracts and cleans YouTube chapter titles", () => {
    expect(extractChapterTitle("02:15 Introduction to Raft")).toBe("Introduction to Raft");
    expect(extractChapterTitle("[10:30] Consensus Protocols")).toBe("Consensus Protocols");
    expect(extractChapterTitle("(01:45:00) System Architecture Overview")).toBe("System Architecture Overview");
    expect(extractChapterTitle("No timestamp chapter")).toBe("No timestamp chapter");
  });

  it("appends watched chapters up to 6 unique headings", () => {
    let headings: string[] = ["00:00 Intro"];
    headings = appendWatchedChapter(headings, "02:15 Raft Overview", 6);
    expect(headings.length).toBe(2);
    expect(headings).toContain("Raft Overview");

    // Duplicate chapter (different case or spacing) -> not added
    headings = appendWatchedChapter(headings, "raft overview", 6);
    expect(headings.length).toBe(2);

    // Add more chapters up to cap of 6
    headings = appendWatchedChapter(headings, "05:00 Leader Election", 6);
    headings = appendWatchedChapter(headings, "08:30 Log Replication", 6);
    headings = appendWatchedChapter(headings, "12:00 Safety Properties", 6);
    headings = appendWatchedChapter(headings, "15:45 Cluster Membership", 6);
    headings = appendWatchedChapter(headings, "19:00 Seventh Chapter Beyond Limit", 6);

    expect(headings.length).toBe(6);
    expect(headings).not.toContain("Seventh Chapter Beyond Limit");
  });
});

describe("11. Article Viewport Reading: 2-Second Dwell Filter", () => {
  it("enforces 2-second viewport dwell threshold before marking elements viewed", () => {
    const t0 = 100000;

    // Fast scroll past element: only in viewport for 800ms (< 2000ms)
    expect(hasMinimumViewportDwell(t0, t0 + 800, 2000)).toBe(false);

    // Barely under 2s: 1999ms -> false
    expect(hasMinimumViewportDwell(t0, t0 + 1999, 2000)).toBe(false);

    // Read attentively: 2500ms (>= 2000ms) -> true!
    expect(hasMinimumViewportDwell(t0, t0 + 2500, 2000)).toBe(true);

    // Edge cases
    expect(hasMinimumViewportDwell(0, 5000, 2000)).toBe(false);
    expect(hasMinimumViewportDwell(t0, t0 - 100, 2000)).toBe(false);
  });

  it("calculates article coverage with 220 WPM reading speed bounded formula", () => {
    // 500 viewed words, 120s dwell, 1000 total words
    // 120s * (220/60) = 440 words bounded
    // min(500, 440) = 440 words verified / 1000 = 0.44
    const cov = calculateArticleCoverage(500, 120, 1000, 220);
    expect(cov).toBe(0.44);
  });
});
