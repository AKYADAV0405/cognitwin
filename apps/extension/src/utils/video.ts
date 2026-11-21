export interface VideoTrackerState {
  videoId: string;
  channelName: string;
  durationSec: number;
  watchedBuckets: Set<number>; // Set of 10s bucket indices (e.g. 0 = 0-10s, 1 = 10-20s)
  playbackRates: number[];
  rewindCount: number;
  pauseCount: number;
  lastPlaybackTime: number;
  isPaused: boolean;
}

const BUCKET_SIZE_SEC = 10;
const REWIND_THRESHOLD_SEC = 5;

/**
 * Calculates unique video coverage percentage based on 10-second timeline buckets.
 * E.g., a 60-second video with 3 unique 10s buckets watched has 50% coverage.
 */
export function calculateVideoCoverage(
  uniqueWatchedBucketsCount: number,
  totalDurationSec: number,
  bucketSizeSec: number = BUCKET_SIZE_SEC
): number {
  if (totalDurationSec <= 0 || uniqueWatchedBucketsCount <= 0) {
    return 0.0;
  }

  const totalPossibleBuckets = Math.ceil(totalDurationSec / bucketSizeSec);
  const rawCoverage = uniqueWatchedBucketsCount / totalPossibleBuckets;
  const clamped = Math.max(0.0, Math.min(1.0, rawCoverage));

  return Math.round(clamped * 1000) / 1000;
}

/**
 * Detects whether a time change constitutes a backward seek (rewind).
 * Threshold is > 5s backward without wrapping from video end to beginning.
 */
export function isRewindSeek(
  previousTimeSec: number,
  currentTimeSec: number,
  thresholdSec: number = REWIND_THRESHOLD_SEC
): boolean {
  if (previousTimeSec <= 0 || currentTimeSec < 0) {
    return false;
  }

  const delta = previousTimeSec - currentTimeSec;
  return delta >= thresholdSec;
}

/**
 * Maps a playback timestamp in seconds to its 10-second bucket index.
 */
export function getBucketIndex(currentTimeSec: number, bucketSizeSec: number = BUCKET_SIZE_SEC): number {
  return Math.floor(Math.max(0, currentTimeSec) / bucketSizeSec);
}
