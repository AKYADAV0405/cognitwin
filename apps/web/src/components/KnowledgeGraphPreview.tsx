import React from "react";
import { Zap, GitCommit, CheckCircle2, AlertTriangle, AlertCircle, Sparkles } from "lucide-react";

export const KnowledgeGraphPreview: React.FC = () => {
  return (
    <div className="relative w-full max-w-xl mx-auto bg-canvas rounded-lg border border-border-subtle shadow-sm overflow-hidden text-ink">
      {/* Window Title Bar */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border-subtle bg-surface/50">
        <div className="flex items-center space-x-2">
          <div className="w-2.5 h-2.5 rounded-full bg-slate-300" />
          <div className="w-2.5 h-2.5 rounded-full bg-slate-300" />
          <div className="w-2.5 h-2.5 rounded-full bg-slate-300" />
        </div>
        <div className="flex items-center space-x-1.5 text-xs font-mono text-ink-muted">
          <GitCommit className="w-3.5 h-3.5 text-cobalt" />
          <span>Knowledge Graph: CS Core (Distributed Systems)</span>
        </div>
        <span className="text-[10px] font-mono uppercase tracking-wider text-ink-subtle bg-surface px-1.5 py-0.5 rounded border border-border-subtle">
          Live Model
        </span>
      </div>

      {/* Simulated Graph Canvas */}
      <div className="relative p-6 bg-surface/30 min-h-[320px] overflow-hidden">
        {/* Subtle mesh pattern inside the card */}
        <div className="absolute inset-0 bg-mesh-pattern pointer-events-none opacity-60" />

        {/* SVG Connecting Edges */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none" xmlns="http://www.w3.org/2000/svg">
          <line x1="140" y1="60" x2="260" y2="60" stroke="#CBD5E1" strokeWidth="1.5" strokeDasharray="3 3" />
          <line x1="260" y1="80" x2="160" y2="170" stroke="#CBD5E1" strokeWidth="1.5" />
          <line x1="260" y1="80" x2="360" y2="170" stroke="#CBD5E1" strokeWidth="1.5" />
          <line x1="160" y1="190" x2="260" y2="250" stroke="#CBD5E1" strokeWidth="1.5" strokeDasharray="3 3" />
        </svg>

        {/* Nodes Grid */}
        <div className="relative z-10 grid grid-cols-2 gap-4">
          {/* Node 1: CAP Theorem (Mastered) */}
          <div className="bg-canvas border border-status-mastered-border rounded-lg p-3 shadow-sm hover:border-status-mastered transition-all">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-semibold text-ink">CAP Theorem</span>
              <span className="inline-flex items-center text-[10px] font-mono font-medium text-status-mastered bg-status-mastered-bg px-1.5 py-0.5 rounded border border-status-mastered-border">
                94%
              </span>
            </div>
            <div className="flex items-center text-[11px] text-status-mastered font-medium">
              <CheckCircle2 className="w-3 h-3 mr-1" />
              <span>Mastered • Anchor Node</span>
            </div>
          </div>

          {/* Node 2: Raft Consensus (Active) */}
          <div className="bg-canvas border border-status-active-border rounded-lg p-3 shadow-sm hover:border-status-active transition-all">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-semibold text-ink">Raft Consensus</span>
              <span className="inline-flex items-center text-[10px] font-mono font-medium text-status-active bg-status-active-bg px-1.5 py-0.5 rounded border border-status-active-border">
                82%
              </span>
            </div>
            <div className="flex items-center text-[11px] text-status-active font-medium">
              <Sparkles className="w-3 h-3 mr-1" />
              <span>Active • High Stability</span>
            </div>
          </div>

          {/* Node 3: Vector Clocks (Fading) */}
          <div className="bg-canvas border border-status-fading-border rounded-lg p-3 shadow-sm hover:border-status-fading transition-all">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-semibold text-ink">Vector Clocks</span>
              <span className="inline-flex items-center text-[10px] font-mono font-medium text-status-fading bg-status-fading-bg px-1.5 py-0.5 rounded border border-status-fading-border">
                41%
              </span>
            </div>
            <div className="flex items-center text-[11px] text-status-fading font-medium">
              <AlertTriangle className="w-3 h-3 mr-1" />
              <span>Fading • Review Due</span>
            </div>
          </div>

          {/* Node 4: Byzantine Faults (At-Risk) */}
          <div className="bg-canvas border border-status-risk-border rounded-lg p-3 shadow-sm hover:border-status-risk transition-all">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-semibold text-ink">Byzantine Faults</span>
              <span className="inline-flex items-center text-[10px] font-mono font-medium text-status-risk bg-status-risk-bg px-1.5 py-0.5 rounded border border-status-risk-border">
                28%
              </span>
            </div>
            <div className="flex items-center text-[11px] text-status-risk font-medium">
              <AlertCircle className="w-3 h-3 mr-1" />
              <span>At Risk • Prerequisite Gap</span>
            </div>
          </div>
        </div>

        {/* Floating Extension Active Badge Overlay */}
        <div className="relative z-20 mt-4 mx-auto max-w-sm bg-canvas/95 backdrop-blur border border-border-strong rounded-full px-3.5 py-1.5 shadow-sm flex items-center justify-between text-xs">
          <div className="flex items-center space-x-2">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-status-mastered opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-status-mastered" />
            </span>
            <span className="font-semibold text-ink">Study Mode: ON</span>
          </div>
          <span className="text-ink-body font-mono text-[11px] truncate max-w-[190px]">
            Capturing: MIT 6.824 Lecture 5
          </span>
        </div>
      </div>

      {/* Card Footer */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-surface border-t border-border-subtle text-[11px] font-mono text-ink-muted">
        <div className="flex items-center space-x-1">
          <GitCommit className="w-3 h-3 text-status-active" />
          <span>24 Active Nodes Modeled</span>
        </div>
        <div className="flex items-center space-x-1 text-ink-subtle">
          <Zap className="w-3 h-3 text-amber-500" />
          <span>GNN Latency: 42ms</span>
        </div>
      </div>
    </div>
  );
};
