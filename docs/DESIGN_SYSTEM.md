# Cognitwin — Design System & Visual Specification

> **Document Version:** 1.0.0-PROD  
> **Aesthetic Classification:** Academic-Modern Editorial Light Mode  
> **Source Reference:** Stitch Production Screens (`881ab6df50a0445ea165f3ef887b2b81` & `6fd57806d3954548b1bfb0a0bb1ffd8f`)  
> **Target Audience:** University Computer Science Students, Academic Researchers, and Self-Taught Software Engineers

---

## 1. Visual Philosophy & Design Principles

Cognitwin balances the precision of an academic research publication with the effortless clarity of modern developer tools (such as Linear light mode, Notion, and Stripe documentation).

1. **Academic-Modern Editorial**:
   - Clean, luminous light canvas. Never use dark cyberpunk neon, glowing laser beams, or gaming UI tropes.
   - Elegant high-contrast serif typography for primary hero headlines, grounded by crisp, neutral sans-serif typography for UI controls and dense data tables.
2. **Precision Engineering**:
   - Crisp 1px geometric borders (`#E2E8F0` / `#CBD5E1`).
   - Maximum 8px corner radius (`rounded-lg`) for cards and dialogs; pills (`rounded-full`) exclusively for status tags and toggle switches.
   - Elevation is communicated via subtle 1px border contrast and soft diffuse shadows (`shadow-sm`), never heavy blurred drop shadows.
3. **Cognitive State Hierarchy**:
   - Colors represent biological and empirical cognitive states: Mastered (Emerald), Active/Stable (Sky), Fading/Decaying (Warm Amber), and Prerequisite Gaps (Crimson).

---

## 2. Color Palette & Semantic Tokens

```
+-----------------------------------------------------------------------------------+
| CANVAS & SURFACE                                                                  |
|   Canvas White:      #FFFFFF   (Primary background)                               |
|   Cool Grey Surface: #F8FAFC   (Section backgrounds, cards, and sidebars)         |
|   Panel Surface:     #F1F5F9   (Subtle nested containers, code blocks)            |
|   Border Default:    #E2E8F0   (Subtle 1px card and table dividers)               |
|   Border Strong:     #CBD5E1   (Interactive inputs, hover states, active tab edges)|
+-----------------------------------------------------------------------------------+
| INK & TYPOGRAPHY                                                                  |
|   Deep Slate Ink:    #0F172A   (Headlines, prominent titles, primary buttons)    |
|   Muted Slate Body:  #475569   (Body copy, descriptions, secondary text)          |
|   Subtle Slate:      #64748B   (Captions, timestamps, inactive icons)             |
|   Muted Border/Line: #94A3B8   (Graph connection edges, disabled controls)        |
+-----------------------------------------------------------------------------------+
| BRAND & COGNITIVE STATUS ACCENTS                                                  |
|   Cobalt Accent:     #1D4ED8   (Brand actions, focus rings, link interactions)    |
|   Cobalt Light Tint: #EFF6FF   (Active hover background, informational banners)   |
|   Emerald Mastered:  #059669   (Mastery >= 85%, verified prerequisite nodes)      |
|   Emerald Light:     #ECFDF5   (Mastered tag background, Study Mode ON status)    |
|   Active / Stable:   #0284C7   (Mastery 60-84%, actively engaged concepts)        |
|   Active Light:      #F0F9FF   (Active tag background)                            |
|   Warm Amber Fading: #D97706   (Decaying concepts, review due, micro-check cue)  |
|   Amber Light:       #FFFBEB   (Warning banner, fading concept node surface)      |
|   Crimson At-Risk:   #DC2626   (Prerequisite gap, error, failed verification)     |
|   Crimson Light:     #FEF2F2   (At-risk concept background)                       |
+-----------------------------------------------------------------------------------+
```

### Tailwind Token Mapping

```javascript
// tailwind.config.js token snippet
module.exports = {
  theme: {
    extend: {
      colors: {
        canvas: '#FFFFFF',
        surface: {
          DEFAULT: '#F8FAFC',
          subtle: '#F1F5F9',
          card: '#FFFFFF',
        },
        ink: {
          DEFAULT: '#0F172A',
          body: '#475569',
          muted: '#64748B',
          subtle: '#94A3B8',
        },
        border: {
          subtle: '#E2E8F0',
          strong: '#CBD5E1',
        },
        cobalt: {
          DEFAULT: '#1D4ED8',
          hover: '#1E40AF',
          surface: '#EFF6FF',
        },
        status: {
          mastered: '#059669',
          'mastered-bg': '#ECFDF5',
          'mastered-border': '#A7F3D0',
          active: '#0284C7',
          'active-bg': '#F0F9FF',
          'active-border': '#BAE6FD',
          fading: '#D97706',
          'fading-bg': '#FFFBEB',
          'fading-border': '#FDE68A',
          risk: '#DC2626',
          'risk-bg': '#FEF2F2',
          'risk-border': '#FECACA',
        }
      },
      borderRadius: {
        lg: '8px',
        md: '6px',
        sm: '4px',
      }
    }
  }
}
```

---

## 3. Signature Graphic Detail: The Knowledge Graph Mesh

To establish visual connection with Graph Neural Networks while maintaining editorial cleanliness, Cognitwin utilizes a **Subtle Light-Mode Wireframe Background**:

- **Characteristics**:
  - Crisp, ultra-thin 1px grid lines and small $2\times 2\text{px}$ dot intersections representing knowledge graph vertices.
  - Rendered at **6% to 10% opacity** in neutral slate (`rgba(15, 23, 42, 0.08)`).
  - Placed behind the Hero Section, Showcase panels, and Authentication views.
- **Fade Fallback & Readability**:
  - The mesh must fade cleanly to solid white `#FFFFFF` behind text blocks using a radial or linear CSS mask:
  ```css
  .bg-mesh-pattern {
    background-image: 
      radial-gradient(circle at 1px 1px, rgba(15, 23, 42, 0.09) 1px, transparent 0),
      linear-gradient(to right, rgba(15, 23, 42, 0.04) 1px, transparent 1px),
      linear-gradient(to bottom, rgba(15, 23, 42, 0.04) 1px, transparent 1px);
    background-size: 24px 24px, 48px 48px, 48px 48px;
    mask-image: radial-gradient(ellipse 70% 60% at 50% 30%, black 20%, transparent 100%);
    -webkit-mask-image: radial-gradient(ellipse 70% 60% at 50% 30%, black 20%, transparent 100%);
  }
  ```

---

## 4. Typography Hierarchy

| Style Role | Font Family | Size / Leading | Weight | Tracking | Usage |
|---|---|---|---|---|---|
| **Display Hero** | Newsreader / Fraunces / Georgia | 52px / 1.15 (`text-5xl`) | 500 (Medium) | `-0.02em` | Main Landing Page Hero ("Your learning, mapped like a living system.") |
| **Section Title** | Inter / Newsreader | 32px / 1.25 (`text-3xl`) | 600 (Semibold) | `-0.015em` | Feature section headlines ("Structured around your actual comprehension.") |
| **Eyebrow / Category** | Inter / JetBrains Mono | 11px / 1.2 (`text-[11px]`) | 700 (Bold) | `+0.08em` uppercase | Section headers (`ENGINEERED PRECISION`, `WORKFLOW & MECHANICS`) |
| **Card Heading** | Inter | 18px / 1.35 (`text-lg`) | 600 (Semibold) | `-0.01em` | Knowledge cards, workflow step titles |
| **Body Standard** | Inter | 15px / 1.6 (`text-[15px]`) | 400 (Regular) | `normal` | Explanatory copy, feature descriptions |
| **Small / Meta** | Inter | 13px / 1.4 (`text-xs`) | 400 / 500 | `normal` | Sub-labels, dates, author tags |
| **Code / Metric** | JetBrains Mono / SFMono | 12px / 1.4 (`text-xs`) | 500 (Medium) | `normal` | Latency readout (`GNN Inference: 42ms`), concept slugs |

---

## 5. Component Specifications (Stitch-Aligned)

### 5.1 Navigation Bar
- **Height**: 64px fixed, backdrop blur with semi-transparent canvas (`bg-white/90 backdrop-blur-md`).
- **Border**: 1px bottom border `#E2E8F0`.
- **Layout**:
  - **Left**: Logo glyph + bold "Cognitwin" typography.
  - **Center**: Nav items (`Overview`, `How It Works`, `Extension`, `Research`) in `#475569` with hover to `#0F172A`.
  - **Right**: "Log In" link + Primary CTA button "Get Started" (Dark Slate `#0F172A`, text `#FFFFFF`, 8px rounded radius).

### 5.2 Hero Showcase Preview Card (The Digital Twin Preview)
- **Container**: White `#FFFFFF` card with 1px border `#E2E8F0`, 8px border radius, soft elevation.
- **Top Bar**: Mac-style window dots, title: `Knowledge Graph: CS Core (Distributed Systems)`.
- **Node Cards (React Flow representation)**:
  - White background, 1px status-colored border, 6px border radius.
  - **Mastered Node** (e.g., *CAP Theorem*): Emerald green tag `94%`, status label `Mastered • Anchor`.
  - **Active Node** (e.g., *Raft Consensus*): Sky blue tag `82%`, status label `Active • Stable`.
  - **Fading Node** (e.g., *Vector Clocks*): Warm amber tag `41%`, status label `Fading • Assessment Due`.
  - **At-Risk Node** (e.g., *Byzantine Faults*): Crimson tag `28%`, status label `Prerequisite Gap`.
- **Floating Badge**:
  - Overlay pill: `Study Mode: ON — Capturing: MIT 6.824 Lecture 5`.
  - Soft green indicator dot with pulse animation.
- **Footer Bar**:
  - `24 Active Nodes Modeled` • `GNN Inference Latency: 42ms`.

### 5.3 3-Card Grid (Engineered Precision)
- **Grid Layout**: 3 columns on desktop, 1 column on mobile.
- **Card Styling**:
  - White background, 1px border `#E2E8F0`, padding 24px.
  - Top-left icon in soft blue/slate container (e.g., passive capture compass, GNN network nodes, quick-check card).
  - Title: 18px semibold `#0F172A`.
  - Description: 14px muted slate `#475569`.
  - Footer pill / tag: Monospace or compact caption with subtle link arrow.

### 5.4 4-Step Horizontal Sequence (Workflow & Mechanics)
- **Step Indicators**:
  - `01 / CAPTURE` — Browse & Study Normally.
  - `02 / PARSE` — Extension Logs Active Focus.
  - `03 / VERIFY` — Quick Verification Check.
  - `04 / UPDATE` — Live Mastery Graph Updates.
- **Layout**: Horizontal card cards with step numbers rendered in bold cobalt/slate monospace uppercase.

### 5.5 Privacy & Academic Integrity Section
- Shield icon badge centered.
- Headline: "Academic integrity and strict privacy by design."
- Feature pills in horizontal row:
  - Checkmark icon + "Explicit toggle switch"
  - Checkmark icon + "Domain allowlist & blocklist"
  - Checkmark icon + "Local-first storage options"

### 5.6 In-App Quick Verification Quiz Component
- Modal or embedded card with high contrast.
- Question prompt rendered in bold `#0F172A` with difficulty badge (e.g., `Difficulty: Medium • Topic: Raft Election`).
- Option cards: 1px border `#E2E8F0` with subtle hover `#F1F5F9`. Selected state: `#1D4ED8` 1.5px border with soft blue tint `#EFF6FF`.
- Timer bar: Discreet 120-second progress bar across top of dialog.

### 5.7 Browser Extension Popup
- Compact dimensions: $340\text{px} \times 460\text{px}$.
- Header: Logo + "Cognitwin Study Mode".
- Prominent Switch: Physical-like pill toggle (Green `#059669` when ON, Slate `#94A3B8` when OFF).
- Active Session Card:
  - Current Domain (e.g., `github.com/cockroachdb/cockroach`).
  - Active Dwell Time counter (e.g., `00:14:32`).
  - Mapped Concept badge (e.g., `Distributed Transactions`).
- Domain Blocklist manager: Input field + quick toggle list for excluding sensitive domains.

---

## 6. Strict UI Constraints & Anti-Patterns

- **NEVER** apply border radius greater than 8px on content containers (cards, modals, inputs). Only pill badges and round toggles may use `rounded-full`.
- **NEVER** use dark neon backgrounds, luminous purple gradients, or cyber glow filters.
- **NEVER** use pure black `#000000` for text; always use Deep Slate Ink `#0F172A`.
- **NEVER** make passive study exposure auto-complete active verification checks; visual distinction between passive dwell and active mastery must remain unambiguous.

---

*This document serves as the visual design and frontend token specification for all Cognitwin user interfaces.*
