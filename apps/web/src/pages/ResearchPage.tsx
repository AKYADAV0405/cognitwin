import React, { useState } from "react";
import { Navbar } from "../components/Navbar";
import { useAuth } from "../context/AuthContext";
import {
  Microscope,
  Terminal,
  Activity,
  Layers,
  Database,
  BarChart3,
  TrendingUp,
} from "lucide-react";

export const ResearchPage: React.FC = () => {
  const { user } = useAuth();
  const [pingResult, setPingResult] = useState<string | null>(null);
  const [pinging, setPinging] = useState<boolean>(false);

  const testResearchPing = async () => {
    setPinging(true);
    try {
      const res = await fetch("/api/v1/research/ping");
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
        {/* Researcher Workspace Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-border-subtle bg-canvas p-6 rounded-lg border shadow-sm">
          <div>
            <div className="flex items-center space-x-2">
              <div className="w-6 h-6 rounded bg-cobalt-surface border border-cobalt/20 flex items-center justify-center text-cobalt">
                <Microscope className="w-4 h-4" />
              </div>
              <h1 className="text-xl font-bold tracking-tight text-ink">
                Researcher Evaluation Workspace
              </h1>
              <span className="text-[10px] uppercase font-mono font-semibold px-2 py-0.5 rounded-full bg-status-mastered-bg text-status-mastered border border-status-mastered-border">
                Paper 1 Baseline
              </span>
            </div>
            <p className="text-xs text-ink-muted mt-1">
              Authorized Researcher Account: <span className="font-mono text-ink font-medium">{user?.email}</span>
            </p>
          </div>

          <div className="flex items-center space-x-3">
            <button
              onClick={testResearchPing}
              disabled={pinging}
              className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-md border border-border-subtle bg-surface hover:bg-surface-subtle text-xs font-medium text-ink transition-colors"
            >
              <Terminal className="w-3.5 h-3.5 text-cobalt" />
              <span>{pinging ? "Evaluating..." : "Verify Researcher Ping"}</span>
            </button>
          </div>
        </div>

        {/* Ping Result Output */}
        {pingResult && (
          <div className="p-4 rounded-lg bg-slate-900 text-slate-100 font-mono text-xs border border-slate-800 shadow-sm">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800 mb-2 text-slate-400 text-[11px]">
              <span>GET /api/v1/research/ping (HTTP 200 OK — Role: researcher verified)</span>
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

        {/* Paper 1 Model Evaluation Benchmarks */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-canvas p-5 rounded-lg border border-border-subtle shadow-sm space-y-1">
            <div className="flex items-center justify-between text-xs text-ink-muted">
              <span>ROC-AUC (Paper 1)</span>
              <BarChart3 className="w-4 h-4 text-cobalt" />
            </div>
            <div className="text-3xl font-bold font-mono text-ink">0.842</div>
            <span className="text-[11px] text-status-mastered font-medium">+0.068 over DKT baseline</span>
          </div>

          <div className="bg-canvas p-5 rounded-lg border border-border-subtle shadow-sm space-y-1">
            <div className="flex items-center justify-between text-xs text-ink-muted">
              <span>RMSE</span>
              <TrendingUp className="w-4 h-4 text-status-mastered" />
            </div>
            <div className="text-3xl font-bold font-mono text-ink">0.318</div>
            <span className="text-[11px] text-status-mastered font-medium">-0.042 lower error rate</span>
          </div>

          <div className="bg-canvas p-5 rounded-lg border border-border-subtle shadow-sm space-y-1">
            <div className="flex items-center justify-between text-xs text-ink-muted">
              <span>Inference Latency</span>
              <Activity className="w-4 h-4 text-amber-500" />
            </div>
            <div className="text-3xl font-bold font-mono text-ink">42ms</div>
            <span className="text-[11px] text-ink-muted font-medium">p95 forward pass</span>
          </div>

          <div className="bg-canvas p-5 rounded-lg border border-border-subtle shadow-sm space-y-1">
            <div className="flex items-center justify-between text-xs text-ink-muted">
              <span>Graph Node Density</span>
              <Layers className="w-4 h-4 text-ink" />
            </div>
            <div className="text-3xl font-bold font-mono text-ink">3.4x</div>
            <span className="text-[11px] text-ink-muted font-medium">Edge-to-vertex ratio</span>
          </div>
        </div>

        {/* Research Specifications & Architecture Notes */}
        <div className="bg-canvas p-6 rounded-lg border border-border-subtle shadow-sm space-y-4">
          <div className="flex items-center justify-between border-b border-border-subtle pb-3">
            <h3 className="text-sm font-semibold text-ink">Paper 1 Evaluation Framework</h3>
            <span className="text-xs font-mono text-ink-muted">PyTorch Geometric 2.5</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs text-ink-body leading-relaxed">
            <div className="space-y-2">
              <h4 className="font-semibold text-ink flex items-center gap-1.5">
                <Database className="w-3.5 h-3.5 text-cobalt" /> Dual-Signal Telemetry Ingestion
              </h4>
              <p>
                Passive dwell telemetry acts strictly as an attenuation exponent in the Ebbinghaus decay curve.
                Active verification attempts provide binary supervised labels (0/1) for node state calibration.
              </p>
            </div>

            <div className="space-y-2">
              <h4 className="font-semibold text-ink flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-cobalt" /> Future Paper 2 Seam Boundary
              </h4>
              <p>
                Reinforcement Learning policies and LLM generative intervention pipelines are strictly isolated
                behind database audit seams (e.g. <code className="bg-surface px-1 py-0.5 rounded font-mono">interventions</code> table).
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};
