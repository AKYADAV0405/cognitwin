const IGNORED_SCHEMES = [
    "chrome:",
    "chrome-extension:",
    "about:",
    "file:",
    "edge:",
    "brave:",
    "view-source:",
    "moz-extension:",
];
const IGNORED_HOSTS = ["localhost:5173", "127.0.0.1:5173"];
/**
 * Sanitizes URLs according to privacy guarantees:
 * - Strips all fragments (#...)
 * - Strips query parameters EXCEPT `v` on youtube.com/watch
 * - Ignores internal browser pages, local development frontend, and YouTube Shorts
 */
export function sanitizeUrl(rawUrl) {
    try {
        const parsed = new URL(rawUrl);
        // 1. Check ignored schemes
        if (IGNORED_SCHEMES.includes(parsed.protocol)) {
            return {
                sanitizedUrl: rawUrl,
                domain: parsed.protocol.replace(":", ""),
                isIgnored: true,
                mediaType: "article",
            };
        }
        // 2. Check ignored hosts (e.g. frontend self-tracking)
        const hostWithPort = parsed.port ? `${parsed.hostname}:${parsed.port}` : parsed.hostname;
        if (IGNORED_HOSTS.includes(hostWithPort)) {
            return {
                sanitizedUrl: parsed.origin,
                domain: parsed.hostname,
                isIgnored: true,
                mediaType: "article",
            };
        }
        // 3. YouTube handling
        const hostnameLower = parsed.hostname.toLowerCase();
        const isYouTube = hostnameLower === "www.youtube.com" ||
            hostnameLower === "youtube.com" ||
            hostnameLower === "m.youtube.com";
        if (isYouTube) {
            if (parsed.pathname.startsWith("/shorts/")) {
                return {
                    sanitizedUrl: `${parsed.origin}${parsed.pathname}`,
                    domain: parsed.hostname,
                    isIgnored: true,
                    mediaType: "video",
                };
            }
            if (parsed.pathname === "/watch") {
                const videoId = parsed.searchParams.get("v");
                if (videoId) {
                    return {
                        sanitizedUrl: `https://www.youtube.com/watch?v=${videoId}`,
                        domain: "youtube.com",
                        isIgnored: false,
                        mediaType: "video",
                    };
                }
            }
            // Ignore non-watch YouTube pages (e.g., feed, explore, subscription lists)
            return {
                sanitizedUrl: `${parsed.origin}${parsed.pathname}`,
                domain: "youtube.com",
                isIgnored: true,
                mediaType: "video",
            };
        }
        // 4. Other video platforms
        if (hostnameLower.includes("vimeo.com") || hostnameLower.includes("youtu.be")) {
            return {
                sanitizedUrl: `${parsed.origin}${parsed.pathname}`,
                domain: parsed.hostname,
                isIgnored: false,
                mediaType: "video",
            };
        }
        // 5. PDF detection
        if (parsed.pathname.toLowerCase().endsWith(".pdf")) {
            return {
                sanitizedUrl: `${parsed.origin}${parsed.pathname}`,
                domain: parsed.hostname,
                isIgnored: false,
                mediaType: "pdf",
            };
        }
        // 6. Docs detection
        const isDocs = hostnameLower.startsWith("docs.") ||
            hostnameLower.startsWith("developer.") ||
            parsed.pathname.includes("/docs/") ||
            parsed.pathname.includes("/documentation/") ||
            parsed.pathname.includes("/api/") ||
            parsed.pathname.includes("/tutorial/") ||
            parsed.pathname.includes("/learn/") ||
            hostnameLower.includes("github.com") ||
            hostnameLower.includes("stackoverflow.com");
        const mediaType = isDocs ? "docs" : "article";
        // Clean standard URL: origin + pathname (no query, no hash)
        // Strip trailing slash for consistency (unless root /)
        let cleanPath = parsed.pathname;
        if (cleanPath.length > 1 && cleanPath.endsWith("/")) {
            cleanPath = cleanPath.slice(0, -1);
        }
        return {
            sanitizedUrl: `${parsed.origin}${cleanPath}`,
            domain: parsed.hostname,
            isIgnored: false,
            mediaType,
        };
    }
    catch {
        return {
            sanitizedUrl: "",
            domain: "",
            isIgnored: true,
            mediaType: "article",
        };
    }
}
