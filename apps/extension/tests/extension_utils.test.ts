import { describe, it, expect } from "vitest";
import { sanitizeUrl } from "../src/utils/url";
import { isBlockedDomain } from "../src/utils/blocklist";
import { calculateArticleCoverage } from "../src/utils/coverage";
import {
  extractKeywords,
  extractHeadings,
  extractKeywordsFromFilename,
  cleanPageTitle,
} from "../src/utils/nlp";
import { checkRevisit, updateRevisitCache } from "../src/utils/revisit";
import { calculateVideoCoverage, isRewindSeek, getBucketIndex } from "../src/utils/video";
import { processSleepGap, coalesceTelemetryEvents, RawTelemetryEvent } from "../src/utils/coalesce";

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
