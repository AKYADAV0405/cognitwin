export interface WarmSessionState {
  sanitizedUrl: string;
  sourceDomain: string;
  pageTitle: string;
  metaDescription?: string | null;
  mediaType: "article" | "docs" | "video" | "pdf";
  startedAt: string;
  activeDwellSeconds: number;
  syncedDwellSeconds: number;
  viewedWords: number;
  totalWords: number;
  viewedHeadings: string[];
  topKeywords: string[];
  watchedBuckets?: number[];
  rewindCount?: number;
  pauseCount?: number;
  clientEventId: string;
  lastUpdatedMs: number;
}

const ONE_HOUR_MS = 60 * 60 * 1000;

/**
 * Calculates unsynced dwell delta between cumulative dwell and already-synced dwell.
 */
export function calculateUnsyncedDwell(activeDwellSeconds: number, syncedDwellSeconds: number): number {
  return Math.max(0, activeDwellSeconds - syncedDwellSeconds);
}

/**
 * Determines whether a session delta meets the threshold to be flushed to the API.
 * The backend enforces active_dwell_seconds >= 15.
 */
export function shouldFlushDelta(
  deltaDwell: number,
  minThreshold: number = 15,
  force: boolean = false
): boolean {
  if (deltaDwell <= 0) return false;
  return force ? deltaDwell >= minThreshold : deltaDwell >= minThreshold;
}

/**
 * Validates whether a cached warm session can be resumed.
 * Checks URL match and ensures session is within the 1-hour window.
 */
export function isWarmSessionValid(
  session: WarmSessionState | null | undefined,
  currentSanitizedUrl: string,
  maxAgeMs: number = ONE_HOUR_MS,
  nowMs: number = Date.now()
): boolean {
  if (!session) return false;
  if (session.sanitizedUrl !== currentSanitizedUrl) return false;
  const age = nowMs - session.lastUpdatedMs;
  return age >= 0 && age < maxAgeMs;
}
