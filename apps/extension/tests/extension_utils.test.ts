import { describe, it, expect } from "vitest";
import { sanitizeUrl } from "../src/utils/url";
import { isBlockedDomain } from "../src/utils/blocklist";
import { calculateArticleCoverage } from "../src/utils/coverage";
import { extractKeywords, extractHeadings, extractKeywordsFromFilename } from "../src/utils/nlp";
import { checkRevisit, updateRevisitCache } from "../src/utils/revisit";
import { calculateVideoCoverage, isRewindSeek, getBucketIndex } from "../src/utils/video";
import { processSleepGap, coalesceTelemetryEvents, RawTelemetryEvent } from "../src/utils/coalesce";

describe("1. URL Sanitization", () => {
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
    // 1000 words total. Read 500 words in 150 seconds (at 200 wpm = 500 max allowed)
    // 500 / 1000 = 0.50
    const cov = calculateArticleCoverage(500, 150, 1000, 200);
    expect(cov).toBe(0.5);
  });

  it("prevents fast-scrolling cheat: dwell time caps viewed words", () => {
    // Fast scrolled through 1000 words in only 15 seconds (15s * 200/60 = 50 words allowed)
    // 50 / 1000 = 0.05
    const cov = calculateArticleCoverage(1000, 15, 1000, 200);
    expect(cov).toBe(0.05);
  });

  it("handles edge cases safely without NaN", () => {
    expect(calculateArticleCoverage(0, 0, 0)).toBe(0.0);
    expect(calculateArticleCoverage(100, 100, -10)).toBe(0.0);
  });
});

describe("4. Term Frequency (TF) Keyword Extraction & Headings", () => {
  it("extracts technical tokens while ignoring stopwords", () => {
    const text = `
      React utilizes the Virtual DOM and hooks like useState and useEffect for state management.
      In distributed consensus, the Raft protocol and Paxos handle fault tolerance and Byzantine faults.
      The quick brown fox jumps over the lazy dog.
    `;
    const keywords = extractKeywords(text, 5);
    expect(keywords.length).toBeLessThanOrEqual(5);
    // Should include high-weight technical terms
    const hasTechTerms = keywords.some((k) =>
      ["react", "virtual", "usestate", "useeffect", "raft", "paxos", "fault"].includes(k)
    );
    expect(hasTechTerms).toBe(true);
  });

  it("sanitizes headings and caps at 5 items <= 100 chars", () => {
    const headings = [
      "  Introduction to Distributed Systems  ",
      "Very long heading ".repeat(10), // > 100 chars
      "Introduction to Distributed Systems", // duplicate
      "CAP Theorem and Consistency",
    ];
    const cleaned = extractHeadings(headings, 5);
    expect(cleaned.length).toBe(3);
    expect(cleaned[0]).toBe("Introduction to Distributed Systems");
    expect(cleaned[1].length).toBeLessThanOrEqual(100);
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

describe("5. 14-Day Local Revisit Cache", () => {
  const now = Date.now();
  const ONE_DAY_MS = 24 * 60 * 60 * 1000;

  it("tags revisit and computes days_since_last_visit", () => {
    const cache = {
      "https://react.dev/learn": {
        last_visited_at: now - 3 * ONE_DAY_MS,
        visit_count: 2,
        max_coverage_pct: 0.6,
      },
    };

    const res = checkRevisit(cache, "https://react.dev/learn", now);
    expect(res.isRevisit).toBe(true);
    expect(res.daysSinceLastVisit).toBe(3);

    const nonExistent = checkRevisit(cache, "https://unknown.com", now);
    expect(nonExistent.isRevisit).toBe(false);
    expect(nonExistent.daysSinceLastVisit).toBeNull();
  });

  it("auto-prunes entries older than 14 days on update", () => {
    const oldCache = {
      "https://old-article.com": {
        last_visited_at: now - 16 * ONE_DAY_MS, // 16 days ago (> 14)
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
    expect(updated["https://old-article.com"]).toBeUndefined(); // Pruned
    expect(updated["https://recent-article.com"]).toBeDefined();
    expect(updated["https://new-article.com"]).toBeDefined();
  });
});

describe("6. YouTube Video Observer & Rewind Counting", () => {
  it("computes coverage from unique 10s timeline buckets", () => {
    // Video is 100 seconds (10 possible 10s buckets)
    // User watched 4 unique buckets (0-10, 10-20, 40-50, 70-80) -> 40% coverage
    const cov = calculateVideoCoverage(4, 100);
    expect(cov).toBe(0.4);
  });

  it("detects rewinds (backward seek > 5s)", () => {
    // User rewinds from 45s to 30s (delta 15s >= 5s threshold) -> true
    expect(isRewindSeek(45, 30)).toBe(true);
    // User rewinds 2s (normal jitter) -> false
    expect(isRewindSeek(45, 43)).toBe(false);
    // User moves forward 45 -> 55 -> false
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
    // Normal 30s checkpoint -> effective delta 30s
    const normal = processSleepGap(t0, t0 + 30000);
    expect(normal.isSleepGap).toBe(false);
    expect(normal.effectiveDeltaSec).toBe(30);

    // 75s delay (capped at 60s)
    const capped = processSleepGap(t0, t0 + 75000);
    expect(capped.isSleepGap).toBe(false);
    expect(capped.effectiveDeltaSec).toBe(60);

    // Laptop slept for 5 minutes (300s > 90s) -> discarded
    const sleep = processSleepGap(t0, t0 + 300000);
    expect(sleep.isSleepGap).toBe(true);
    expect(sleep.effectiveDeltaSec).toBe(0);
  });

  it("drops visits < 15s and coalesces visits in the same hour", () => {
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
        viewed_headings: ["Hooks"],
        top_keywords: ["hooks", "state"],
        is_revisit: false,
        days_since_last_visit: null,
        started_at: "2026-11-21T10:20:00Z",
        ended_at: "2026-11-21T10:21:00Z",
      },
    ];

    const coalesced = coalesceTelemetryEvents(events);
    expect(coalesced.length).toBe(1); // evt-1 dropped (<15s), evt-2 and evt-3 coalesced
    expect(coalesced[0].active_dwell_seconds).toBe(105); // 45 + 60
    expect(coalesced[0].estimated_coverage_pct).toBe(0.7); // max(0.3, 0.7)
    expect(coalesced[0].viewed_headings).toContain("Intro");
    expect(coalesced[0].viewed_headings).toContain("Hooks");
    expect(coalesced[0].top_keywords).toContain("state");
  });
});
