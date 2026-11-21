import React, { useEffect, useState, useCallback } from "react";
import { Navbar } from "../components/Navbar";
import { useAuth } from "../context/AuthContext";
import { KnowledgeGraphPreview } from "../components/KnowledgeGraphPreview";
import {
  Activity,
  CheckCircle2,
  Clock,
  Play,
  Terminal,
  Key,
  Copy,
  Check,
  Download,
  RefreshCw,
  Trash2,
  FileText,
  BookOpen,
  Video,
  FileCode,
  RotateCcw,
  Info,
  ShieldCheck,
  ExternalLink,
} from "lucide-react";

interface ExtensionTokenStatus {
  id: string;
  name: string;
  last_used_at: string | null;
  created_at: string;
  is_active: boolean;
}

interface PassiveExposure {
  id: string;
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
  video_metrics: {
    channel_name?: string;
    video_duration_sec?: number;
    playback_rate_avg?: number;
    rewind_count: number;
    pause_count: number;
  } | null;
  started_at: string;
  ended_at: string;
  received_at: string;
}

export const DashboardPage: React.FC = () => {
  const { user } = useAuth();
  const [pingResult, setPingResult] = useState<string | null>(null);
  const [pinging, setPinging] = useState<boolean>(false);

  // Extension Token State
  const [tokenStatus, setTokenStatus] = useState<ExtensionTokenStatus[]>([]);
  const [newToken, setNewToken] = useState<string | null>(null);
  const [copied, setCopied] = useState<boolean>(false);
  const [tokenLoading, setTokenLoading] = useState<boolean>(false);
  const [tokenError, setTokenError] = useState<string | null>(null);

  // Passive Telemetry State
  const [recentExposures, setRecentExposures] = useState<PassiveExposure[]>([]);
  const [telemetryLoading, setTelemetryLoading] = useState<boolean>(false);

  const fetchTokenStatus = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/auth/extension-token/status");
      if (res.ok) {
        const data = await res.json();
        setTokenStatus(data);
      }
    } catch {
      // Ignore
    }
  }, []);

  const fetchRecentTelemetry = useCallback(async () => {
    setTelemetryLoading(true);
    try {
      const res = await fetch("/api/v1/telemetry/passive/recent");
      if (res.ok) {
        const data = await res.json();
        setRecentExposures(data);
      }
    } catch {
      // Ignore
    } finally {
      setTelemetryLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchTokenStatus();
    fetchRecentTelemetry();
  }, [fetchTokenStatus, fetchRecentTelemetry]);

  const generateToken = async () => {
    setTokenLoading(true);
    setTokenError(null);
    try {
      const res = await fetch("/api/v1/auth/extension-token", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: "Chrome Extension" }),
      });
      if (!res.ok) throw new Error("Failed to generate extension token.");
      const data = await res.json();
      setNewToken(data.token);
      await fetchTokenStatus();
    } catch (err) {
      setTokenError(err instanceof Error ? err.message : "Failed to generate token.");
    } finally {
      setTokenLoading(false);
    }
  };

  const revokeTokens = async () => {
    if (!window.confirm("Are you sure you want to revoke all extension tokens? Connected extensions will stop syncing until reconnected.")) {
      return;
    }
    setTokenLoading(true);
    try {
      await fetch("/api/v1/auth/extension-token", { method: "DELETE" });
      setNewToken(null);
      await fetchTokenStatus();
    } finally {
      setTokenLoading(false);
    }
  };

  const copyToClipboard = () => {
    if (!newToken) return;
    navigator.clipboard.writeText(newToken);
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  const testStudentPing = async () => {
    setPinging(true);
    try {
      const res = await fetch("/api/v1/student/ping");
      const data = await res.json();
      setPingResult(JSON.stringify(data, null, 2));
    } catch (err) {
      setPingResult(err instanceof Error ? err.message : "Ping failed");
    } finally {
      setPinging(false);
    }
  };

  const formatDwellTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (mins === 0) return `${secs}s`;
    return `${mins}m ${secs}s`;
  };

  const hasActiveToken = tokenStatus.some((t) => t.is_active);

  return (
    <div className="min-h-screen bg-surface flex flex-col text-ink">
      <Navbar />

      <main className="flex-1 max-w-6xl w-full mx-auto px-6 py-8 space-y-8">
        {/* Welcome Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-border-subtle bg-canvas p-6 rounded-lg border shadow-sm">
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-xl font-bold tracking-tight text-ink">
                Student Workspace
              </h1>
              <span className="text-[10px] uppercase font-mono font-semibold px-2 py-0.5 rounded-full bg-status-active-bg text-status-active border border-status-active-border">
                {user?.role} Account
              </span>
            </div>
            <p className="text-xs text-ink-muted mt-1">
              Signed in as <span className="font-mono text-ink font-medium">{user?.email}</span> via {user?.auth_provider} authentication.
            </p>
          </div>

          <div className="flex items-center space-x-3">
            <button
              onClick={testStudentPing}
              disabled={pinging}
              className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-md border border-border-subtle bg-surface hover:bg-surface-subtle text-xs font-medium text-ink transition-colors"
            >
              <Terminal className="w-3.5 h-3.5 text-cobalt" />
              <span>{pinging ? "Pinging..." : "Verify RBAC Ping"}</span>
            </button>
          </div>
        </div>

        {/* Ping Result Output */}
        {pingResult && (
          <div className="p-4 rounded-lg bg-slate-900 text-slate-100 font-mono text-xs border border-slate-800 shadow-sm">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800 mb-2 text-slate-400 text-[11px]">
              <span>GET /api/v1/student/ping (HTTP 200 OK)</span>
              <button
                onClick={() => setPingResult(null)}
                className="hover:text-white"
              >
                ✕
              </button>
            </div>
            <pre className="overflow-x-auto">{pingResult}</pre>
          </div>
        )}

        {/* Metric Summary Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-canvas p-4 rounded-lg border border-border-subtle shadow-sm space-y-1">
            <div className="flex items-center justify-between text-xs text-ink-muted">
              <span>Mastered Concepts</span>
              <CheckCircle2 className="w-4 h-4 text-status-mastered" />
            </div>
            <div className="text-2xl font-bold font-mono text-ink">8</div>
            <span className="text-[11px] text-status-mastered font-medium">≥ 85% probability</span>
          </div>

          <div className="bg-canvas p-4 rounded-lg border border-border-subtle shadow-sm space-y-1">
            <div className="flex items-center justify-between text-xs text-ink-muted">
              <span>Active Concepts</span>
              <Activity className="w-4 h-4 text-status-active" />
            </div>
            <div className="text-2xl font-bold font-mono text-ink">12</div>
            <span className="text-[11px] text-status-active font-medium">Stable retention</span>
          </div>

          <div className="bg-canvas p-4 rounded-lg border border-border-subtle shadow-sm space-y-1">
            <div className="flex items-center justify-between text-xs text-ink-muted">
              <span>Review Due</span>
              <Clock className="w-4 h-4 text-status-fading" />
            </div>
            <div className="text-2xl font-bold font-mono text-ink">3</div>
            <span className="text-[11px] text-status-fading font-medium">Forgetting threshold</span>
          </div>

          <div className="bg-canvas p-4 rounded-lg border border-border-subtle shadow-sm space-y-1">
            <div className="flex items-center justify-between text-xs text-ink-muted">
              <span>Passive Exposures</span>
              <ShieldCheck className="w-4 h-4 text-cobalt" />
            </div>
            <div className="text-2xl font-bold font-mono text-ink">{recentExposures.length}</div>
            <span className="text-[11px] text-cobalt font-medium">Dual-signal synced</span>
          </div>
        </div>

        {/* ==================================================================== */}
        {/* Phase 3: Chrome Extension Connection Card */}
        {/* ==================================================================== */}
        <div className="bg-canvas rounded-lg border border-border-subtle p-6 shadow-sm space-y-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-border-subtle">
            <div className="flex items-center space-x-3">
              <div className="w-9 h-9 rounded-lg bg-surface border border-border-subtle flex items-center justify-center text-ink">
                <Key className="w-4 h-4 text-cobalt" />
              </div>
              <div>
                <h2 className="text-sm font-semibold tracking-tight text-ink">
                  Chrome Extension Passive Telemetry Connection
                </h2>
                <p className="text-xs text-ink-muted">
                  Connect your privacy-first browser extension with a scoped Bearer token.
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <span
                className={`inline-flex items-center text-[11px] font-mono font-medium px-2.5 py-0.5 rounded-full border ${
                  hasActiveToken
                    ? "bg-status-mastered-bg text-status-mastered border-status-mastered-border"
                    : "bg-surface-subtle text-ink-muted border-border-subtle"
                }`}
              >
                {hasActiveToken ? "● Connected Token Active" : "○ Disconnected"}
              </span>

              {hasActiveToken && (
                <button
                  onClick={revokeTokens}
                  disabled={tokenLoading}
                  className="inline-flex items-center space-x-1 px-2.5 py-1 text-xs text-status-risk hover:bg-status-risk-bg rounded-md border border-transparent transition-colors"
                  title="Revoke all active extension tokens"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>Revoke</span>
                </button>
              )}
            </div>
          </div>

          {/* Action Row */}
          {!newToken && (
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <p className="text-xs text-ink-body max-w-xl leading-relaxed">
                Tokens are stored as SHA-256 hashes on the server. When generated, copy the token and paste it directly into your Chrome Extension popup.
              </p>
              <div className="flex items-center gap-2 shrink-0">
                <a
                  href="/downloads/cognitwin-extension.zip"
                  download="cognitwin-extension.zip"
                  className="inline-flex items-center space-x-1.5 px-3 py-2 rounded-md border border-border-subtle bg-surface hover:bg-surface-subtle text-xs font-medium text-ink transition-colors"
                >
                  <Download className="w-3.5 h-3.5 text-cobalt" />
                  <span>Download Extension (.zip)</span>
                </a>
                <button
                  onClick={generateToken}
                  disabled={tokenLoading}
                  className="inline-flex items-center space-x-1.5 px-4 py-2 rounded-md bg-ink text-canvas hover:bg-ink-body text-xs font-semibold transition-colors disabled:opacity-50"
                >
                  <Key className="w-3.5 h-3.5" />
                  <span>{tokenLoading ? "Generating..." : hasActiveToken ? "Regenerate Token" : "Generate Extension Token"}</span>
                </button>
              </div>
            </div>
          )}

          {/* New Token Banner */}
          {newToken && (
            <div className="p-4 rounded-lg bg-status-mastered-bg border border-status-mastered-border space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-status-mastered flex items-center gap-1.5">
                  <CheckCircle2 className="w-4 h-4" /> Extension Token Generated
                </span>
                <span className="text-[11px] font-mono text-ink-muted">Copy now (shown once)</span>
              </div>
              <div className="flex items-center space-x-2">
                <input
                  type="text"
                  readOnly
                  value={newToken}
                  className="flex-1 px-3 py-2 text-xs font-mono bg-canvas border border-status-mastered-border rounded-md text-ink select-all"
                />
                <button
                  onClick={copyToClipboard}
                  className="inline-flex items-center space-x-1.5 px-3 py-2 rounded-md bg-ink text-canvas hover:bg-ink-body text-xs font-semibold transition-colors shrink-0"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copied ? "Copied!" : "Copy"}</span>
                </button>
              </div>
            </div>
          )}

          {tokenError && (
            <p className="text-xs text-status-risk">{tokenError}</p>
          )}

          {/* Quick Load Unpacked Instructions */}
          <div className="p-4 rounded-lg bg-surface border border-border-subtle text-xs text-ink-body space-y-2">
            <div className="flex items-center space-x-1.5 font-semibold text-ink">
              <Info className="w-3.5 h-3.5 text-cobalt" />
              <span>How to load in Google Chrome:</span>
            </div>
            <ol className="list-decimal list-inside space-y-1 text-ink-muted pl-1">
              <li>Open Google Chrome and navigate to <span className="font-mono text-ink">chrome://extensions</span></li>
              <li>Toggle <strong className="text-ink font-medium">Developer mode</strong> ON in the top-right corner.</li>
              <li>Click <strong className="text-ink font-medium">Load unpacked</strong> and select the directory: <code className="bg-canvas px-1.5 py-0.5 rounded border border-border-subtle font-mono text-ink">apps/extension/dist</code></li>
              <li>Click the Cognitwin puzzle icon in your toolbar, paste your Bearer token, and toggle <strong className="text-status-mastered font-medium">Study Mode ON</strong>!</li>
            </ol>
          </div>
        </div>

        {/* ==================================================================== */}
        {/* Phase 3: Recent Passive Study Activity Table */}
        {/* ==================================================================== */}
        <div className="bg-canvas rounded-lg border border-border-subtle p-6 shadow-sm space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-border-subtle">
            <div>
              <h2 className="text-sm font-semibold tracking-tight text-ink">
                Recent Passive Study Telemetry
              </h2>
              <p className="text-xs text-ink-muted">
                Dwell times and reading coverage logged by your Chrome Extension.
              </p>
            </div>
            <button
              onClick={fetchRecentTelemetry}
              disabled={telemetryLoading}
              className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-md border border-border-subtle bg-surface hover:bg-surface-subtle text-xs font-medium text-ink transition-colors"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${telemetryLoading ? "animate-spin" : ""}`} />
              <span>Refresh</span>
            </button>
          </div>

          {recentExposures.length === 0 ? (
            <div className="text-center py-10 space-y-2">
              <BookOpen className="w-8 h-8 text-ink-subtle mx-auto opacity-50" />
              <p className="text-xs font-medium text-ink">No passive study sessions synced yet</p>
              <p className="text-[11px] text-ink-muted max-w-sm mx-auto">
                Turn Study Mode ON in your browser extension and read technical documentation or watch video tutorials to record focus telemetry.
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-border-subtle text-[11px] text-ink-muted uppercase font-mono tracking-wider">
                    <th className="pb-2.5 font-medium">Source / Document</th>
                    <th className="pb-2.5 font-medium">Media</th>
                    <th className="pb-2.5 font-medium">Dwell</th>
                    <th className="pb-2.5 font-medium">Coverage</th>
                    <th className="pb-2.5 font-medium">Keywords & Context</th>
                    <th className="pb-2.5 font-medium text-right">Synced</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border-subtle">
                  {recentExposures.map((item) => (
                    <tr key={item.id} className="hover:bg-surface-subtle transition-colors">
                      <td className="py-3 pr-4 max-w-xs">
                        <div className="font-medium text-ink truncate" title={item.page_title}>
                          {item.page_title}
                        </div>
                        <div className="text-[11px] text-ink-muted font-mono flex items-center gap-1 truncate">
                          <span>{item.source_domain}</span>
                          <a
                            href={item.source_url}
                            target="_blank"
                            rel="noreferrer"
                            className="hover:text-cobalt inline-flex items-center"
                          >
                            <ExternalLink className="w-2.5 h-2.5" />
                          </a>
                        </div>
                      </td>

                      <td className="py-3 pr-4 whitespace-nowrap">
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-medium capitalize bg-surface border border-border-subtle text-ink-body">
                          {item.media_type === "video" && <Video className="w-3 h-3 text-status-fading" />}
                          {item.media_type === "docs" && <BookOpen className="w-3 h-3 text-cobalt" />}
                          {item.media_type === "article" && <FileText className="w-3 h-3 text-emerald-600" />}
                          {item.media_type === "pdf" && <FileCode className="w-3 h-3 text-purple-600" />}
                          {item.media_type}
                        </span>
                      </td>

                      <td className="py-3 pr-4 whitespace-nowrap font-mono text-ink-body">
                        {formatDwellTime(item.active_dwell_seconds)}
                      </td>

                      <td className="py-3 pr-4 whitespace-nowrap">
                        <div className="flex items-center space-x-2">
                          <div className="w-16 h-1.5 bg-surface border border-border-subtle rounded-full overflow-hidden">
                            <div
                              className="h-full bg-status-mastered rounded-full"
                              style={{ width: `${Math.round(item.estimated_coverage_pct * 100)}%` }}
                            />
                          </div>
                          <span className="font-mono text-[11px] text-ink-body">
                            {Math.round(item.estimated_coverage_pct * 100)}%
                          </span>
                        </div>
                      </td>

                      <td className="py-3 pr-4 max-w-xs">
                        <div className="flex flex-wrap gap-1">
                          {item.is_revisit && (
                            <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-semibold bg-status-fading-bg text-status-fading border border-status-fading-border">
                              <RotateCcw className="w-2.5 h-2.5 mr-0.5" /> Revisit ({item.days_since_last_visit}d)
                            </span>
                          )}
                          {item.video_metrics?.rewind_count ? (
                            <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-semibold bg-purple-50 text-purple-700 border border-purple-200">
                              {item.video_metrics.rewind_count} rewinds
                            </span>
                          ) : null}
                          {item.top_keywords.slice(0, 3).map((kw, i) => (
                            <span
                              key={i}
                              className="inline-block px-1.5 py-0.5 rounded text-[9px] font-mono bg-surface text-ink-muted border border-border-subtle"
                            >
                              {kw}
                            </span>
                          ))}
                        </div>
                      </td>

                      <td className="py-3 text-right whitespace-nowrap font-mono text-[11px] text-ink-muted">
                        {new Date(item.ended_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Main Content Grid: Knowledge Graph and Seams */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Knowledge Graph Card (8 Cols) */}
          <div className="lg:col-span-8 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold tracking-tight text-ink">
                Digital Twin Knowledge Graph
              </h2>
              <span className="text-xs text-ink-muted font-mono">
                Model: GNN-KT v1.0
              </span>
            </div>
            <KnowledgeGraphPreview />
          </div>

          {/* Side Panel: Active Quick Verification Teaser (Phase 4 Seam) */}
          <div className="lg:col-span-4 space-y-6">
            {/* Quick Check Card */}
            <div className="bg-canvas p-5 rounded-lg border border-border-subtle shadow-sm space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-ink">Active Quick Check</span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-surface text-ink-muted border border-border-subtle">
                  Phase 4 Engine
                </span>
              </div>
              <p className="text-xs text-ink-body leading-relaxed">
                Vector Clocks has reached an elevated forgetting risk (41%). Complete a 2-minute micro-check to calibrate node embeddings.
              </p>
              <div className="pt-2">
                <button
                  disabled
                  className="w-full flex items-center justify-center space-x-1.5 px-3 py-2 rounded-md bg-surface border border-border-subtle text-xs text-ink-muted font-medium cursor-not-allowed"
                >
                  <Play className="w-3.5 h-3.5" />
                  <span>Quiz Engine Seeds in Phase 4</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};
