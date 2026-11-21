export const COMMON_STOPWORDS = new Set([
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

export const GENERIC_EXAMPLE_NOUNS = new Set([
  "example", "examples", "sample", "samples", "tutorial", "tutorials",
  "step", "steps", "method", "methods", "class", "classes",
  "code", "codes", "image", "images", "output", "outputs",
  "input", "inputs", "figure", "figures", "table", "tables",
  "section", "sections", "guide", "guides", "overview", "introduction",
  "summary", "conclusion", "result", "results", "file", "files",
  "line", "lines", "item", "items", "page", "pages", "click",
  "follow", "read", "write", "note", "notes", "tip", "tips",
  "exercise", "exercises", "solution", "solutions", "source", "author",
  "question", "questions", "answer", "answers", "video", "description",
]);

/**
 * Strips site-name suffixes and branding delimiters from page titles.
 * E.g.: "Data Preprocessing in Python - GeeksforGeeks" -> "Data Preprocessing in Python"
 *       "Array.prototype.map() - JavaScript | MDN" -> "Array.prototype.map() - JavaScript"
 */
export function cleanPageTitle(rawTitle: string): string {
  if (!rawTitle || !rawTitle.trim()) return "Untitled Document";

  let title = rawTitle.trim();

  // Strip leading notification counts like "(2) "
  title = title.replace(/^\(\d+\)\s*/, "");

  // Strip known high-profile brand suffixes
  const knownBrandPattern =
    /\s+[-|—·::]\s+(GeeksforGeeks|MDN Web Docs|MDN|YouTube|Stack Overflow|Wikipedia|W3Schools|GitHub|Medium|Dev\.to|LeetCode|Python documentation|Kaggle|Coursera|edX).*$/i;
  if (knownBrandPattern.test(title)) {
    title = title.replace(knownBrandPattern, "");
  } else {
    // Strip general trailing delimiter suffix: "Title - SiteName"
    const genericSuffixPattern = /\s+[-|—·::]\s+[^-|—·::]{2,35}$/;
    if (genericSuffixPattern.test(title)) {
      title = title.replace(genericSuffixPattern, "").trim();
    }
  }

  return title.slice(0, 160).trim() || rawTitle.slice(0, 160).trim();
}

export interface ExtractionOptions {
  title?: string;
  headings?: string[];
  codeSnippets?: string[];
  maxKeywords?: number;
}

/**
 * High-signal keyword extractor:
 * - Weights title, headings, and <code> tags 3x higher than body text.
 * - Extracts and prioritizes 2-word technical bigrams (e.g. "data preprocessing", "machine learning").
 * - Filters generic example nouns ("example", "tutorial", "step", "method", etc.).
 */
export function extractKeywords(
  bodyText: string,
  optionsOrMaxKeywords: ExtractionOptions | number = 10
): string[] {
  const options: ExtractionOptions =
    typeof optionsOrMaxKeywords === "number"
      ? { maxKeywords: optionsOrMaxKeywords }
      : optionsOrMaxKeywords;

  const maxKeywords = options.maxKeywords || 10;
  const tokenRegex = /[a-zA-Z0-9_\-#+]{2,40}/g;

  const scoreMap = new Map<string, number>();

  function processText(text: string, weightMultiplier: number) {
    if (!text || !text.trim()) return;

    const matches = text.match(tokenRegex) || [];
    const validTokens: string[] = [];

    // 1. Process unigrams
    for (const rawToken of matches) {
      const lower = rawToken.toLowerCase();

      // Check if valid keyword candidate
      const isStopword = COMMON_STOPWORDS.has(lower);
      const isGeneric = GENERIC_EXAMPLE_NOUNS.has(lower);
      const isPureNumber = /^\d+$/.test(lower);

      if (!isStopword && !isGeneric && !isPureNumber && lower.length >= 3) {
        validTokens.push(lower);

        // Technical bonus for camelCase, snake_case, C++ symbols
        const isTechToken =
          /[A-Z]/.test(rawToken.slice(1)) ||
          rawToken.includes("_") ||
          rawToken.includes("-") ||
          rawToken.includes("#") ||
          rawToken.includes("+");

        const tokenScore = (isTechToken ? 2.0 : 1.0) * weightMultiplier;
        scoreMap.set(lower, (scoreMap.get(lower) || 0) + tokenScore);
      } else {
        validTokens.push(""); // Boundary break for bigrams
      }
    }

    // 2. Extract technical 2-word bigrams
    for (let i = 0; i < validTokens.length - 1; i++) {
      const w1 = validTokens[i];
      const w2 = validTokens[i + 1];

      if (w1 && w2) {
        const bigram = `${w1} ${w2}`;
        // Bigrams get 2.5x bonus score over unigrams
        const bigramScore = 2.5 * weightMultiplier;
        scoreMap.set(bigram, (scoreMap.get(bigram) || 0) + bigramScore);
      }
    }
  }

  // A. Title: weight 3.0x
  if (options.title) {
    processText(options.title, 3.0);
  }

  // B. Headings: weight 3.0x
  if (options.headings && options.headings.length > 0) {
    for (const h of options.headings) {
      processText(h, 3.0);
    }
  }

  // C. Code snippets: weight 3.0x
  if (options.codeSnippets && options.codeSnippets.length > 0) {
    for (const c of options.codeSnippets) {
      processText(c, 3.0);
    }
  }

  // D. Main body text: weight 1.0x
  processText(bodyText, 1.0);

  // Sort by weighted score descending
  const sorted = Array.from(scoreMap.entries())
    .sort((a, b) => b[1] - a[1])
    .map(([phrase]) => phrase);

  // Keep top unique keywords/bigrams
  return sorted.slice(0, maxKeywords);
}

/**
 * Sanitizes and extracts at most `maxHeadings` strings, each <= 100 characters.
 */
export function extractHeadings(headings: string[], maxHeadings: number = 6): string[] {
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
    const cleanTitle = decodeURIComponent(withoutExt).replace(/[_-]+/g, " ").trim();

    const title = cleanPageTitle(cleanTitle ? cleanTitle.slice(0, 160) : "PDF Document");
    const keywords = extractKeywords(title, { title, maxKeywords: 10 });

    return { title, keywords };
  } catch {
    return { title: "PDF Document", keywords: [] };
  }
}
