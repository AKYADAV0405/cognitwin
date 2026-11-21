import React from "react";
import { Link } from "react-router-dom";
import { Navbar } from "../components/Navbar";
import { KnowledgeGraphPreview } from "../components/KnowledgeGraphPreview";
import {
  Compass,
  Network,
  CheckSquare,
  ArrowRight,
  ShieldCheck,
  CheckCircle2,
  Lock,
  Database,
  ExternalLink,
} from "lucide-react";

export const LandingPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-canvas text-ink flex flex-col">
      <Navbar />

      <main className="flex-1">
        {/* ==================================================================== */}
        {/* Hero Section with Knowledge Graph Mesh Background */}
        {/* ==================================================================== */}
        <section className="relative pt-12 pb-20 md:pt-20 md:pb-28 overflow-hidden">
          {/* Subtle Wireframe Lattice Mesh (6-10% opacity) */}
          <div className="absolute inset-0 bg-mesh-pattern pointer-events-none z-0" />

          <div className="relative z-10 max-w-6xl mx-auto px-6">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
              {/* Left Column: Editorial Headline & Copy */}
              <div className="lg:col-span-6 space-y-6">
                {/* Academic Version Pill */}
                <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-surface border border-border-subtle text-[11px] font-mono text-ink-muted">
                  <span className="w-1.5 h-1.5 rounded-full bg-cobalt" />
                  <span>Research-Backed Learning System • v1.0</span>
                </div>

                {/* Display Hero Headline in Editorial Serif */}
                <h1 className="font-editorial text-4xl sm:text-5xl lg:text-[54px] font-medium leading-[1.12] tracking-tight text-ink">
                  Your learning, mapped like a living system.
                </h1>

                {/* Subheading */}
                <p className="text-base sm:text-lg text-ink-body leading-relaxed font-normal">
                  Cognitwin connects a toggleable Chrome extension with Graph Neural Network mastery
                  tracking to model your cognitive state as you learn tech skills online.
                </p>

                {/* Call-to-Actions */}
                <div className="flex flex-col sm:flex-row items-stretch sm:items-center space-y-3 sm:space-y-0 sm:space-x-4 pt-2">
                  <Link
                    to="/signup"
                    className="inline-flex items-center justify-center px-5 py-2.5 rounded-md bg-ink text-canvas text-xs font-semibold hover:bg-ink-body transition-colors shadow-sm"
                  >
                    <span>Start Free</span>
                    <ArrowRight className="w-3.5 h-3.5 ml-2" />
                  </Link>
                  <a
                    href="#how-it-works"
                    className="inline-flex items-center justify-center px-4 py-2.5 rounded-md border border-border-subtle text-ink-body text-xs font-medium hover:bg-surface transition-colors"
                  >
                    <span>See how the extension works</span>
                    <ArrowRight className="w-3 h-3 ml-1.5 text-ink-muted" />
                  </a>
                </div>

                {/* Social Proof */}
                <div className="pt-2 text-xs text-ink-muted flex items-center space-x-2">
                  <span className="text-slate-400">🎓</span>
                  <span>Built for university CS students & self-taught engineers</span>
                </div>
              </div>

              {/* Right Column: Stitch Knowledge Graph Preview Card */}
              <div className="lg:col-span-6">
                <KnowledgeGraphPreview />
              </div>
            </div>
          </div>
        </section>

        {/* ==================================================================== */}
        {/* Section 2: Engineered Precision */}
        {/* ==================================================================== */}
        <section id="overview" className="py-20 border-t border-border-subtle bg-surface/40">
          <div className="max-w-6xl mx-auto px-6">
            <div className="max-w-2xl mb-14">
              <span className="text-[11px] font-mono font-bold uppercase tracking-wider text-cobalt">
                ENGINEERED PRECISION
              </span>
              <h2 className="font-editorial text-3xl sm:text-4xl font-medium tracking-tight text-ink mt-2 mb-3">
                Structured around your actual comprehension.
              </h2>
              <p className="text-sm text-ink-body leading-relaxed">
                Designed specifically for independent learning environments where passive video
                consumption rarely translates to production readiness.
              </p>
            </div>

            {/* 3-Card Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Card 1: Passive Study Capture */}
              <div className="bg-canvas border border-border-subtle rounded-lg p-6 shadow-sm flex flex-col justify-between hover:border-border-strong transition-all">
                <div className="space-y-3">
                  <div className="w-10 h-10 rounded-lg bg-surface border border-border-subtle flex items-center justify-center text-cobalt">
                    <Compass className="w-5 h-5" />
                  </div>
                  <h3 className="text-base font-semibold text-ink">Passive Study Capture</h3>
                  <p className="text-xs text-ink-body leading-relaxed">
                    A light, toggleable browser extension built for technical YouTube lectures,
                    GitHub documentation, and arXiv whitepapers. It ignores social media, distraction
                    sites, and idle tabs by design.
                  </p>
                </div>
                <div className="mt-6 pt-4 border-t border-border-subtle text-[11px] font-mono text-ink-muted flex items-center justify-between">
                  <span>Background process when toggled off</span>
                  <Lock className="w-3.5 h-3.5 text-ink-subtle" />
                </div>
              </div>

              {/* Card 2: Graph-Based Mastery */}
              <div className="bg-canvas border border-border-subtle rounded-lg p-6 shadow-sm flex flex-col justify-between hover:border-border-strong transition-all">
                <div className="space-y-3">
                  <div className="w-10 h-10 rounded-lg bg-surface border border-border-subtle flex items-center justify-center text-cobalt">
                    <Network className="w-5 h-5" />
                  </div>
                  <h3 className="text-base font-semibold text-ink">Graph-Based Mastery</h3>
                  <p className="text-xs text-ink-body leading-relaxed">
                    A living Knowledge Graph driven by Graph Neural Networks that understands
                    prerequisite relationships and models forgetting curves before technical
                    interviews or exams.
                  </p>
                </div>
                <div className="mt-6 pt-4 border-t border-border-subtle text-[11px] font-mono text-ink-muted flex items-center justify-between">
                  <span>Prerequisite-aware decay curve</span>
                  <ExternalLink className="w-3.5 h-3.5 text-ink-subtle" />
                </div>
              </div>

              {/* Card 3: Targeted Quick Checks */}
              <div className="bg-canvas border border-border-subtle rounded-lg p-6 shadow-sm flex flex-col justify-between hover:border-border-strong transition-all">
                <div className="space-y-3">
                  <div className="w-10 h-10 rounded-lg bg-surface border border-border-subtle flex items-center justify-center text-cobalt">
                    <CheckSquare className="w-5 h-5" />
                  </div>
                  <h3 className="text-base font-semibold text-ink">Targeted Quick Checks</h3>
                  <p className="text-xs text-ink-body leading-relaxed">
                    No bloated 50-card flashcard decks. Cognitwin creates 2-minute micro-checks
                    exclusively at critical conceptual junctions where your cognitive model registers
                    uncertainty.
                  </p>
                </div>
                <div className="mt-6 pt-4 border-t border-border-subtle text-[11px] font-mono text-ink-muted flex items-center justify-between">
                  <span>Avg. 90s per session</span>
                  <CheckCircle2 className="w-3.5 h-3.5 text-status-mastered" />
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ==================================================================== */}
        {/* Section 3: Workflow & Mechanics (4 Steps) */}
        {/* ==================================================================== */}
        <section id="how-it-works" className="py-20 border-t border-border-subtle bg-canvas">
          <div className="max-w-6xl mx-auto px-6">
            <div className="max-w-2xl mb-14">
              <span className="text-[11px] font-mono font-bold uppercase tracking-wider text-ink-muted">
                WORKFLOW & MECHANICS
              </span>
              <h2 className="font-editorial text-3xl sm:text-4xl font-medium tracking-tight text-ink mt-2 mb-3">
                From raw browsing to structured retention in 4 steps.
              </h2>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
              {/* Step 1 */}
              <div className="bg-surface rounded-lg p-5 border border-border-subtle flex flex-col justify-between">
                <div>
                  <span className="text-[11px] font-mono font-bold text-cobalt">01 / CAPTURE</span>
                  <h4 className="text-sm font-semibold text-ink mt-2 mb-1.5">Browse & Study Normally</h4>
                  <p className="text-xs text-ink-body leading-relaxed">
                    Read technical documentation, watch architecture breakdowns, or inspect source code
                    with Study Mode activated in your browser.
                  </p>
                </div>
                <span className="mt-4 pt-3 border-t border-border-subtle text-[10px] font-mono text-ink-muted">
                  passive focus tracking
                </span>
              </div>

              {/* Step 2 */}
              <div className="bg-surface rounded-lg p-5 border border-border-subtle flex flex-col justify-between">
                <div>
                  <span className="text-[11px] font-mono font-bold text-cobalt">02 / PARSE</span>
                  <h4 className="text-sm font-semibold text-ink mt-2 mb-1.5">Extension Logs Active Focus</h4>
                  <p className="text-xs text-ink-body leading-relaxed">
                    The extension model measures user engagement concepts, algorithmic snippets, and active
                    dwell time without storing page text.
                  </p>
                </div>
                <span className="mt-4 pt-3 border-t border-border-subtle text-[10px] font-mono text-ink-muted">
                  privacy-first metadata
                </span>
              </div>

              {/* Step 3 */}
              <div className="bg-surface rounded-lg p-5 border border-border-subtle flex flex-col justify-between">
                <div>
                  <span className="text-[11px] font-mono font-bold text-cobalt">03 / VERIFY</span>
                  <h4 className="text-sm font-semibold text-ink mt-2 mb-1.5">Quick Verification Check</h4>
                  <p className="text-xs text-ink-body leading-relaxed">
                    When cognitive courage peaks, the system delivers a targeted 3-question verification
                    to anchor foundational prerequisite knowledge.
                  </p>
                </div>
                <span className="mt-4 pt-3 border-t border-border-subtle text-[10px] font-mono text-ink-muted">
                  micro-check triggered
                </span>
              </div>

              {/* Step 4 */}
              <div className="bg-surface rounded-lg p-5 border border-border-subtle flex flex-col justify-between">
                <div>
                  <span className="text-[11px] font-mono font-bold text-cobalt">04 / UPDATE</span>
                  <h4 className="text-sm font-semibold text-ink mt-2 mb-1.5">Live Mastery Graph Updates</h4>
                  <p className="text-xs text-ink-body leading-relaxed">
                    Your Digital Twin recalibrates its cognitive edge weights, scheduling optimal spaced
                    intervals and highlighting structural blind spots.
                  </p>
                </div>
                <span className="mt-4 pt-3 border-t border-border-subtle text-[10px] font-mono text-ink-muted">
                  weights updated
                </span>
              </div>
            </div>
          </div>
        </section>

        {/* ==================================================================== */}
        {/* Section 4: Privacy & Academic Integrity */}
        {/* ==================================================================== */}
        <section id="extension" className="py-20 border-t border-border-subtle bg-surface/50">
          <div className="max-w-4xl mx-auto px-6 text-center space-y-6">
            <div className="w-12 h-12 rounded-full bg-canvas border border-border-subtle flex items-center justify-center mx-auto text-cobalt shadow-sm">
              <ShieldCheck className="w-6 h-6" />
            </div>

            <h2 className="font-editorial text-3xl sm:text-4xl font-medium tracking-tight text-ink">
              Academic integrity and strict privacy by design.
            </h2>

            <p className="text-sm sm:text-base text-ink-body max-w-2xl mx-auto leading-relaxed">
              Study Mode is 100% opt-in. You decide when Cognitwin observes. Built-in domain blocklist,
              zero cross-site ad tracking, and complete data deletion at any click.
            </p>

            <div className="flex flex-wrap items-center justify-center gap-3 pt-4">
              <div className="inline-flex items-center space-x-2 px-3 py-1.5 rounded-full bg-canvas border border-border-subtle text-xs text-ink font-medium shadow-sm">
                <CheckCircle2 className="w-3.5 h-3.5 text-status-mastered" />
                <span>Explicit toggle switch</span>
              </div>
              <div className="inline-flex items-center space-x-2 px-3 py-1.5 rounded-full bg-canvas border border-border-subtle text-xs text-ink font-medium shadow-sm">
                <CheckCircle2 className="w-3.5 h-3.5 text-status-mastered" />
                <span>Domain allowlist & blocklist</span>
              </div>
              <div className="inline-flex items-center space-x-2 px-3 py-1.5 rounded-full bg-canvas border border-border-subtle text-xs text-ink font-medium shadow-sm">
                <Database className="w-3.5 h-3.5 text-status-mastered" />
                <span>Local-first storage options</span>
              </div>
            </div>
          </div>
        </section>

        {/* ==================================================================== */}
        {/* Section 5: Bottom CTA Band */}
        {/* ==================================================================== */}
        <section className="py-20 border-t border-border-subtle bg-canvas">
          <div className="max-w-4xl mx-auto px-6">
            <div className="bg-surface border border-border-subtle rounded-lg p-8 sm:p-12 shadow-sm text-center space-y-4">
              <span className="text-[11px] font-mono font-bold uppercase tracking-wider text-cobalt">
                GET STARTED
              </span>
              <h2 className="font-editorial text-3xl sm:text-4xl font-medium tracking-tight text-ink">
                Ready to map your tech stack mastery?
              </h2>
              <p className="text-sm text-ink-body max-w-xl mx-auto">
                Connect the Chrome extension and start tracking your distributed systems, cloud, or
                algorithm learning curves today.
              </p>
              <div className="pt-2">
                <Link
                  to="/signup"
                  className="inline-flex items-center justify-center px-6 py-3 rounded-md bg-ink text-canvas text-xs font-semibold hover:bg-ink-body transition-colors shadow-sm"
                >
                  <span>Get Started with Cognitwin</span>
                  <ArrowRight className="w-3.5 h-3.5 ml-2" />
                </Link>
                <p className="text-[11px] text-ink-muted mt-3">
                  No credit card required • Free for students
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* ==================================================================== */}
      {/* Footer */}
      {/* ==================================================================== */}
      <footer className="border-t border-border-subtle bg-canvas py-12 text-xs text-ink-body">
        <div className="max-w-6xl mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 pb-10">
            {/* Column 1: Info */}
            <div className="space-y-3 md:col-span-1">
              <div className="flex items-center space-x-2 font-bold text-ink">
                <div className="w-5 h-5 rounded bg-ink flex items-center justify-center text-canvas text-[10px]">
                  C
                </div>
                <span>Cognitwin</span>
              </div>
              <p className="text-[11px] text-ink-muted leading-relaxed">
                An adaptive cognitive digital twin architecture engineered for focused learning,
                empirical evaluation, and continuous cognitive modeling.
              </p>
            </div>

            {/* Column 2: Architecture */}
            <div>
              <h5 className="font-mono text-[11px] uppercase tracking-wider font-semibold text-ink mb-3">
                Architecture
              </h5>
              <ul className="space-y-2 text-[11px] text-ink-muted">
                <li><a href="#how-it-works" className="hover:text-ink">How It Works</a></li>
                <li><Link to="/research" className="hover:text-ink">Paper 1 Architecture</Link></li>
                <li><a href="#extension" className="hover:text-ink">Browser Extension</a></li>
              </ul>
            </div>

            {/* Column 3: Resources */}
            <div>
              <h5 className="font-mono text-[11px] uppercase tracking-wider font-semibold text-ink mb-3">
                Resources
              </h5>
              <ul className="space-y-2 text-[11px] text-ink-muted">
                <li><a href="/api/v1/docs" className="hover:text-ink">API Documentation</a></li>
                <li><Link to="/research" className="hover:text-ink">Research Benchmarks</Link></li>
                <li><a href="https://github.com" target="_blank" rel="noreferrer" className="hover:text-ink">GitHub Repository</a></li>
              </ul>
            </div>

            {/* Column 4: System & Legal */}
            <div>
              <h5 className="font-mono text-[11px] uppercase tracking-wider font-semibold text-ink mb-3">
                System & Legal
              </h5>
              <ul className="space-y-2 text-[11px] text-ink-muted">
                <li><span className="hover:text-ink cursor-pointer">Privacy Policy</span></li>
                <li><span className="hover:text-ink cursor-pointer">Terms of Service</span></li>
                <li><Link to="/login" className="hover:text-ink">Student Portal</Link></li>
              </ul>
            </div>
          </div>

          <div className="border-t border-border-subtle pt-6 flex flex-col sm:flex-row items-center justify-between text-[11px] text-ink-subtle">
            <span>© 2026 Cognitwin. An Adaptive Cognitive Digital Twin System.</span>
            <span className="font-mono">v1.0.0-prod</span>
          </div>
        </div>
      </footer>
    </div>
  );
};
