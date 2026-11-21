export const DEFAULT_PRIVACY_BLOCKLIST: string[] = [
  "mail.google.com",
  "web.whatsapp.com",
  "instagram.com",
  "x.com",
  "twitter.com",
  "facebook.com",
  "netflix.com",
  "tiktok.com",
  "reddit.com",
  "linkedin.com",
  "chase.com",
  "bankofamerica.com",
  "wellsfargo.com",
  "paypal.com",
  "stripe.com",
  "fidelity.com",
  "schwab.com",
];

/**
 * Checks if a domain or URL matches the active privacy blocklist.
 * Matches exact hostnames or subdomains (e.g. `mail.google.com` or `*.bankofamerica.com`).
 */
export function isBlockedDomain(domainOrUrl: string, blocklist: string[] = DEFAULT_PRIVACY_BLOCKLIST): boolean {
  if (!domainOrUrl) return false;

  let host = domainOrUrl.toLowerCase().trim();
  try {
    if (host.includes("://")) {
      const url = new URL(host);
      host = url.hostname.toLowerCase();
    }
  } catch {
    // If not a full URL, treat as hostname string
  }

  // Strip port if present
  if (host.includes(":")) {
    host = host.split(":")[0];
  }

  for (const blocked of blocklist) {
    const cleanBlocked = blocked.toLowerCase().trim();
    if (!cleanBlocked) continue;

    if (host === cleanBlocked || host.endsWith(`.${cleanBlocked}`)) {
      return true;
    }
  }

  return false;
}
