export interface RawTelemetryEvent {
  client_event_id: string;
  source_domain: string;
  source_url: string;
  page_title: string;
  media_type: "article" | "docs" | "video" | "pdf";
  active_dwell_seconds: number;
  estimated_coverage_pct: number;
  viewed_headings: string[];
  top_keywords: string[];
  is_revisit: boolean;
  days_since_last_visit: number | null;
  video_metrics?: {
    channel_name?: string;
    video_duration_sec?: number;
    playback_rate_avg?: number;
    rewind_count: number;
    pause_count: number;
  } | null;
  started_at: string;
  ended_at: string;
}

/**
 * Evaluates the time delta between two consecutive focus checkpoints.
 *
 * Rules:
 * - If delta > 90 seconds (gap caused by laptop sleep or OS suspension),
 *   the entire gap is discarded (`isSleepGap = true`, `effectiveDeltaSec = 0`).
 * - Otherwise, any single checkpoint delta is capped at 60 seconds max.
 */
export function processSleepGap(
  lastCheckMs: number,
  currentCheckMs: number
): { effectiveDeltaSec: number; isSleepGap: boolean } {
  if (lastCheckMs <= 0 || currentCheckMs <= lastCheckMs) {
    return { effectiveDeltaSec: 0, isSleepGap: false };
  }

  const rawDeltaSec = (currentCheckMs - lastCheckMs) / 1000;

  // Sleep gap threshold: > 90s gap is discarded completely
  if (rawDeltaSec > 90) {
    return { effectiveDeltaSec: 0, isSleepGap: true };
  }

  // Cap single checkpoint delta at 60s
  const cappedDeltaSec = Math.min(60, Math.max(0, Math.round(rawDeltaSec)));
  return { effectiveDeltaSec: cappedDeltaSec, isSleepGap: false };
}

/**
 * Gets a discrete hour bucket string (e.g., '2026-11-21T23:00') for coalescing.
 */
export function getHourBucket(isoDateString: string): string {
  try {
    const d = new Date(isoDateString);
    return `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, "0")}-${String(
      d.getUTCDate()
    ).padStart(2, "0")}T${String(d.getUTCHours()).padStart(2, "0")}:00Z`;
  } catch {
    return "unknown-bucket";
  }
}

/**
 * Filters out visits < 20s and coalesces rapid tab-hopping visits to the same
 * URL within the same hour bucket into a single unified event.
 */
export function coalesceTelemetryEvents(events: RawTelemetryEvent[]): RawTelemetryEvent[] {
  // 1. Drop visits < 20s
  const validEvents = events.filter((e) => e.active_dwell_seconds >= 20);
  if (!validEvents.length) return [];

  const grouped = new Map<string, RawTelemetryEvent>();

  for (const event of validEvents) {
    const hourBucket = getHourBucket(event.started_at);
    const groupKey = `${event.source_url}::${hourBucket}`;

    const existing = grouped.get(groupKey);
    if (!existing) {
      grouped.set(groupKey, {
        ...event,
        viewed_headings: [...event.viewed_headings],
        top_keywords: [...event.top_keywords],
      });
    } else {
      // Coalesce:
      // Sum dwell seconds
      existing.active_dwell_seconds += event.active_dwell_seconds;
      // Max coverage
      existing.estimated_coverage_pct = Math.max(
        existing.estimated_coverage_pct,
        event.estimated_coverage_pct
      );
      // Merge unique headings (up to 5)
      const mergedHeadings = Array.from(
        new Set([...existing.viewed_headings, ...event.viewed_headings])
      ).slice(0, 5);
      existing.viewed_headings = mergedHeadings;

      // Merge unique keywords (up to 10)
      const mergedKeywords = Array.from(
        new Set([...existing.top_keywords, ...event.top_keywords])
      ).slice(0, 10);
      existing.top_keywords = mergedKeywords;

      // Update ended_at to the latest timestamp
      if (new Date(event.ended_at) > new Date(existing.ended_at)) {
        existing.ended_at = event.ended_at;
      }
      // Revisit flag is true if either is revisit
      existing.is_revisit = existing.is_revisit || event.is_revisit;

      // Merge video metrics if present
      if (event.video_metrics && existing.video_metrics) {
        existing.video_metrics.rewind_count += event.video_metrics.rewind_count;
        existing.video_metrics.pause_count += event.video_metrics.pause_count;
      } else if (event.video_metrics && !existing.video_metrics) {
        existing.video_metrics = { ...event.video_metrics };
      }
    }
  }

  return Array.from(grouped.values());
}
