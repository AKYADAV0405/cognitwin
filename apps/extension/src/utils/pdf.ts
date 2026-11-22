import { calculateArticleCoverage } from "./coverage";
import { extractKeywordsFromFilename, cleanPageTitle } from "./nlp";

export interface PdfSessionState {
  tabId: number;
  sanitizedUrl: string;
  sourceDomain: string;
  pageTitle: string;
  mediaType: "pdf";
  topKeywords: string[];
  activeDwellSeconds: number;
  syncedDwellSeconds: number;
  lastHeartbeatMs: number;
  startedAt: string;
  clientEventId: string;
}

/**
 * Initializes a new background PDF tracking session.
 */
export function createPdfSession(
  tabId: number,
  sanitizedUrl: string,
  domain: string,
  tabTitle?: string,
  nowIso: string = new Date().toISOString(),
  nowMs: number = Date.now(),
  customEventId?: string
): PdfSessionState {
  const { title: filenameTitle, keywords: filenameKeywords } = extractKeywordsFromFilename(sanitizedUrl);
  const rawTitle = tabTitle ? tabTitle.replace(/\.pdf$/i, "").trim() : filenameTitle;
  const pageTitle = cleanPageTitle(rawTitle || "PDF Document");
  const keywords = filenameKeywords.length > 0 ? filenameKeywords : [pageTitle.toLowerCase().slice(0, 30)];

  const eventId =
    customEventId ||
    (typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID()
      : `pdf-${nowMs}-${Math.random().toString(36).slice(2, 8)}`);

  return {
    tabId,
    sanitizedUrl,
    sourceDomain: domain,
    pageTitle,
    mediaType: "pdf",
    topKeywords: keywords.slice(0, 8),
    activeDwellSeconds: 0,
    syncedDwellSeconds: 0,
    lastHeartbeatMs: nowMs,
    startedAt: nowIso,
    clientEventId: eventId,
  };
}

/**
 * Calculates reading coverage for PDF documents bounded by 220 WPM reading speed
 * assuming a standard academic paper / whitepaper length of ~2500 words.
 */
export function calculatePdfCoverage(
  dwellSeconds: number,
  estimatedWords: number = 2500,
  wordsPerMinute: number = 220
): number {
  const estimatedReadWords = dwellSeconds * (wordsPerMinute / 60);
  return calculateArticleCoverage(
    estimatedReadWords,
    dwellSeconds,
    estimatedWords,
    wordsPerMinute
  );
}
