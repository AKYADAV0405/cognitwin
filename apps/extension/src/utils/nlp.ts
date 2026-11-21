const COMMON_STOPWORDS = new Set([
  "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
  "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
  "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
  "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
  "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
  "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
  "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
  "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
  "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
  "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
  "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
  "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
  "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
  "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
  "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
  "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
  "yourselves", "will", "shall", "may", "might", "can", "must", "also", "just", "use", "using",
  "used", "page", "website", "click", "home", "search", "like", "get", "new", "one", "two",
]);

/**
 * Lightweight rule-based Term Frequency (TF) keyword extractor.
 * Strips common stopwords, rewards technical tokens (camelCase, snake_case, PascalCase, symbols),
 * and returns at most `maxKeywords` unique terms.
 *
 * Implements strict privacy: raw page text is consumed locally and discarded immediately.
 */
export function extractKeywords(text: string, maxKeywords: number = 10): string[] {
  if (!text || !text.trim()) return [];

  // Match words, code identifiers (camelCase, snake_case, kebab-case, C++ style)
  const tokenRegex = /[a-zA-Z0-9_\-#+]{2,40}/g;
  const matches = text.match(tokenRegex) || [];

  const freqMap = new Map<string, number>();

  for (const rawToken of matches) {
    const lower = rawToken.toLowerCase();

    // Skip stopwords or purely numeric tokens
    if (COMMON_STOPWORDS.has(lower) || /^\d+$/.test(lower)) {
      continue;
    }

    // Weight boost for technical identifiers (camelCase or underscores or tech symbols)
    const isTechToken =
      /[A-Z]/.test(rawToken.slice(1)) ||
      rawToken.includes("_") ||
      rawToken.includes("-") ||
      rawToken.includes("#") ||
      rawToken.includes("+");

    const weight = isTechToken ? 2.5 : 1.0;
    freqMap.set(lower, (freqMap.get(lower) || 0) + weight);
  }

  // Sort by weighted frequency descending
  const sorted = Array.from(freqMap.entries())
    .sort((a, b) => b[1] - a[1])
    .map(([word]) => word);

  return sorted.slice(0, maxKeywords);
}

/**
 * Sanitizes and extracts at most `maxHeadings` strings, each <= 100 characters.
 */
export function extractHeadings(headings: string[], maxHeadings: number = 5): string[] {
  if (!headings || !headings.length) return [];

  const unique = new Set<string>();
  const result: string[] = [];

  for (const h of headings) {
    const clean = h.replace(/\s+/g, " ").trim();
    if (!clean || clean.length < 3) continue;

    const truncated = clean.length > 100 ? clean.slice(0, 97) + "..." : clean;
    const lower = truncated.toLowerCase();

    if (!unique.has(lower)) {
      unique.add(lower);
      result.push(truncated);
      if (result.length >= maxHeadings) break;
    }
  }

  return result;
}

/**
 * Extracts page title and keywords from a PDF URL or file name without DOM scraping.
 */
export function extractKeywordsFromFilename(filenameOrPath: string): { title: string; keywords: string[] } {
  try {
    const rawName = filenameOrPath.split("/").pop() || filenameOrPath;
    const withoutExt = rawName.replace(/\.pdf$/i, "");
    // Replace underscores, hyphens, and %20 with spaces
    const cleanTitle = decodeURIComponent(withoutExt).replace(/[_-]+/g, " ").trim();

    const title = cleanTitle ? cleanTitle.slice(0, 160) : "PDF Document";
    const keywords = extractKeywords(cleanTitle, 10);

    return { title, keywords };
  } catch {
    return { title: "PDF Document", keywords: [] };
  }
}
