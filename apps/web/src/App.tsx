import React, { useEffect, useState } from "react";
import { Activity, CheckCircle2, RefreshCw, Server, AlertCircle } from "lucide-react";

interface HealthData {
  status: string;
  database: string;
  version: string;
  environment: string;
}

export const App: React.FC = () => {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastChecked, setLastChecked] = useState<string>("");

  const checkHealth = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/v1/health");
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      const data: HealthData = await res.json();
      setHealth(data);
      setLastChecked(new Date().toLocaleTimeString());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to connect to API service");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkHealth();
  }, []);

  return (
    <div className="min-h-screen bg-surface flex flex-col items-center justify-center p-6 text-ink">
      <div className="w-full max-w-md bg-canvas rounded-lg border border-border-subtle p-6 shadow-sm">
        <div className="flex items-center justify-between pb-4 border-b border-border-subtle mb-6">
          <div className="flex items-center space-x-2">
            <Activity className="w-5 h-5 text-cobalt" />
            <h1 className="text-lg font-semibold tracking-tight text-ink">Cognitwin Core</h1>
          </div>
          <span className="text-[11px] font-mono font-medium uppercase tracking-wider text-ink-muted bg-surface-subtle px-2 py-0.5 rounded border border-border-subtle">
            Phase 1 Skeleton
          </span>
        </div>

        <div className="space-y-4">
          <div className="bg-surface-subtle rounded-md p-4 border border-border-subtle">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium text-ink-body">API Health Status</span>
              {loading ? (
                <span className="text-xs text-ink-muted flex items-center gap-1">
                  <RefreshCw className="w-3 h-3 animate-spin" /> Checking...
                </span>
              ) : health?.status === "healthy" ? (
                <span className="inline-flex items-center gap-1 text-xs font-medium text-status-mastered bg-status-mastered-bg px-2 py-0.5 rounded-full border border-status-mastered-border">
                  <CheckCircle2 className="w-3 h-3" /> Healthy
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 text-xs font-medium text-status-risk bg-status-risk-bg px-2 py-0.5 rounded-full border border-status-risk-border">
                  <AlertCircle className="w-3 h-3" /> Degraded
                </span>
              )}
            </div>

            {error ? (
              <p className="text-xs text-status-risk mt-1">{error}</p>
            ) : health ? (
              <div className="text-xs space-y-1 font-mono text-ink-body">
                <div className="flex justify-between">
                  <span className="text-ink-muted">Database:</span>
                  <span className="text-status-mastered font-medium">{health.database}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-ink-muted">Version:</span>
                  <span>{health.version}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-ink-muted">Environment:</span>
                  <span>{health.environment}</span>
                </div>
              </div>
            ) : null}
          </div>

          <div className="flex items-center justify-between text-xs text-ink-muted pt-2">
            <span>Last checked: {lastChecked || "Never"}</span>
            <button
              onClick={checkHealth}
              disabled={loading}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-ink text-canvas hover:bg-ink-body transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`w-3 h-3 ${loading ? "animate-spin" : ""}`} />
              Re-check API
            </button>
          </div>
        </div>

        <div className="mt-6 pt-4 border-t border-border-subtle flex items-center justify-between text-[11px] text-ink-subtle">
          <span className="flex items-center gap-1">
            <Server className="w-3 h-3" /> Vite Proxy: /api/* &rarr; :8000
          </span>
          <span>v1.0.0-phase1</span>
        </div>
      </div>
    </div>
  );
};

export default App;
