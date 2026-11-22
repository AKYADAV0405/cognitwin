/**
 * Calculates reading-speed bounded article coverage percentage.
 *
 * Formula:
 * estimated_coverage_pct = min(1.0, min(viewed_words, active_dwell_seconds * (words_per_minute / 60)) / total_article_words)
 *
 * Guarantees that simply fast-scrolling to the bottom without reading dwell time
 * will NOT falsely report 100% coverage.
 */
export function calculateArticleCoverage(
  viewedWords: number,
  activeDwellSeconds: number,
  totalArticleWords: number,
  wordsPerMinute: number = 220
): number {
  if (totalArticleWords <= 0 || activeDwellSeconds <= 0 || viewedWords <= 0) {
    return 0.0;
  }

  const wordsPerSecond = wordsPerMinute / 60;
  const dwellBoundedWords = activeDwellSeconds * wordsPerSecond;
  const verifiedWords = Math.min(viewedWords, dwellBoundedWords);

  const rawCoverage = verifiedWords / totalArticleWords;
  const clampedCoverage = Math.max(0.0, Math.min(1.0, rawCoverage));

  // Round to 3 decimal places
  return Math.round(clampedCoverage * 1000) / 1000;
}

/**
 * Checks if an element has continuously remained visible in the viewport
 * for at least the specified duration (default 2000ms / 2 seconds).
 * Prevents fast-scrolling past elements from falsely marking them viewed.
 */
export function hasMinimumViewportDwell(
  enteredAtMs: number,
  checkedAtMs: number,
  minDurationMs: number = 2000
): boolean {
  if (enteredAtMs <= 0 || checkedAtMs < enteredAtMs) {
    return false;
  }
  return checkedAtMs - enteredAtMs >= minDurationMs;
}
