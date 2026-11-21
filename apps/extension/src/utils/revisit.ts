export interface RevisitEntry {
  last_visited_at: number; // Unix timestamp in ms
  visit_count: number;
  max_coverage_pct: number;
}

export type RevisitCache = Record<string, RevisitEntry>;

const FOURTEEN_DAYS_MS = 14 * 24 * 60 * 60 * 1000;
const MAX_CACHE_ENTRIES = 500;

/**
 * Checks if a sanitized URL has been visited within the 14-day window.
 */
export function checkRevisit(
  cache: RevisitCache,
  sanitizedUrl: string,
  nowMs: number = Date.now()
): { isRevisit: boolean; daysSinceLastVisit: number | null } {
  const entry = cache[sanitizedUrl];
  if (!entry) {
    return { isRevisit: false, daysSinceLastVisit: null };
  }

  const diffMs = nowMs - entry.last_visited_at;
  if (diffMs > FOURTEEN_DAYS_MS || diffMs < 0) {
    return { isRevisit: false, daysSinceLastVisit: null };
  }

  const days = Math.round((diffMs / (24 * 60 * 60 * 1000)) * 10) / 10;
  return {
    isRevisit: true,
    daysSinceLastVisit: days,
  };
}

/**
 * Updates the 14-day revisit cache for a visited URL, auto-prunes entries
 * older than 14 days, and evicts oldest items if cache exceeds 500 entries.
 */
export function updateRevisitCache(
  cache: RevisitCache,
  sanitizedUrl: string,
  coveragePct: number,
  nowMs: number = Date.now()
): RevisitCache {
  const nextCache: RevisitCache = {};

  // 1. Copy and auto-prune entries older than 14 days
  for (const [url, entry] of Object.entries(cache)) {
    if (nowMs - entry.last_visited_at <= FOURTEEN_DAYS_MS) {
      nextCache[url] = { ...entry };
    }
  }

  // 2. Insert or update the current entry
  const existing = nextCache[sanitizedUrl];
  if (existing) {
    nextCache[sanitizedUrl] = {
      last_visited_at: nowMs,
      visit_count: existing.visit_count + 1,
      max_coverage_pct: Math.max(existing.max_coverage_pct, coveragePct),
    };
  } else {
    nextCache[sanitizedUrl] = {
      last_visited_at: nowMs,
      visit_count: 1,
      max_coverage_pct: coveragePct,
    };
  }

  // 3. Enforce maximum 500 entries (LRU-like eviction of oldest `last_visited_at`)
  const entries = Object.entries(nextCache);
  if (entries.length > MAX_CACHE_ENTRIES) {
    entries.sort((a, b) => a[1].last_visited_at - b[1].last_visited_at);
    const toRemoveCount = entries.length - MAX_CACHE_ENTRIES;
    for (let i = 0; i < toRemoveCount; i++) {
      delete nextCache[entries[i][0]];
    }
  }

  return nextCache;
}
