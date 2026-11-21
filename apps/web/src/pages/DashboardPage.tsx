import React, { useState } from "react";
import { Navbar } from "../components/Navbar";
import { useAuth } from "../context/AuthContext";
import { KnowledgeGraphPreview } from "../components/KnowledgeGraphPreview";
import {
  Activity,
  CheckCircle2,
  Clock,
  AlertTriangle,
  Play,
  Download,
  Terminal,
} from "lucide-react";

export const DashboardPage: React.FC = () => {
  const { user } = useAuth();
  const [pingResult, setPingResult] = useState<string | null>(null);
  const [pinging, setPinging] = useState<boolean>(false);

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
              <span>Prerequisite Gaps</span>
              <AlertTriangle className="w-4 h-4 text-status-risk" />
            </div>
            <div className="text-2xl font-bold font-mono text-ink">1</div>
            <span className="text-[11px] text-status-risk font-medium">Requires verification</span>
          </div>
        </div>

        {/* Main Content Grid */}
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

          {/* Side Panel: Study Mode & Quick Verification Teaser (4 Cols) */}
          <div className="lg:col-span-4 space-y-6">
            {/* Extension Status Card */}
            <div className="bg-canvas p-5 rounded-lg border border-border-subtle shadow-sm space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-ink">Chrome Extension</span>
                <span className="inline-flex items-center text-[10px] font-mono font-medium text-status-mastered bg-status-mastered-bg px-2 py-0.5 rounded-full border border-status-mastered-border">
                  Ready (Phase 3)
                </span>
              </div>
              <p className="text-xs text-ink-body leading-relaxed">
                Study Mode automatically measures focused dwell time on arXiv, GitHub, and YouTube without ingesting personal browsing data.
              </p>
              <div className="pt-2">
                <button
                  disabled
                  className="w-full flex items-center justify-center space-x-1.5 px-3 py-2 rounded-md bg-surface border border-border-subtle text-xs text-ink-muted font-medium cursor-not-allowed"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Extension Loading in Phase 3</span>
                </button>
              </div>
            </div>

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
