---
name: ui-design
description: Applies a Cursor-inspired dark UI design system (layered charcoal surfaces, single accent, 4px grid, typography scale, components, a11y). Use when building or polishing frontend UI, components, pages, dashboards, admin panels, data views, or when the user wants modern, clean, beautiful, or dark aesthetics.
---

# UI Design — Cursor-Inspired Design System

## When to activate

- Building any frontend component, page, or layout
- Refactoring UI to look more modern
- Requests for “pretty”, “clean”, “modern”, or “dark”
- Dashboards, tools, admin panels, data views

## Philosophy

- Deep layered dark backgrounds — warm charcoal, not pure black
- Subtle surface hierarchy — small elevation steps between panels
- One accent color — CTAs and focus only
- Restraint — whitespace over decoration
- Content-first — chrome recedes; content leads

---

## Color system (dark default)

```css
/* Backgrounds — deepest → highest */
--bg-base:        #0d0d0f;
--bg-surface:     #141416;
--bg-elevated:    #1c1c1f;
--bg-overlay:     #242428;
--bg-input:       #1a1a1d;

/* Borders */
--border-subtle:  #ffffff0f;
--border-default: #ffffff1a;
--border-strong:  #ffffff2e;

/* Text */
--text-primary:   #f0f0f2;
--text-secondary: #8b8b96;
--text-muted:     #4a4a55;
--text-inverse:   #0d0d0f;

/* Accent — one per project */
--accent:         #7c6af7;
--accent-hover:   #9181f8;
--accent-muted:   #7c6af720;

/* Semantic — status only */
--success:        #3dd68c;
--warning:        #f5a623;
--error:          #f87171;
--info:           #60a5fa;
```

### Color rules

- Never use pure `#000000` or `#ffffff`; use tokens above
- At least three background levels: base → surface → elevated
- Accent on ~≤10% of UI: primary actions, active nav, focus
- Semantic colors for status indicators only, not decoration

---

## Typography

```css
font-family: 'Inter', 'Geist', system-ui, -apple-system, sans-serif;

--text-xs:   11px;   /* badges, timestamps */
--text-sm:   13px;   /* secondary body */
--text-base: 14px;   /* primary UI */
--text-lg:   16px;   /* section headers */
--text-xl:   20px;   /* page titles */
--text-2xl:  28px;   /* hero */

--font-regular:  400;
--font-medium:   500;
--font-semibold: 600;

--leading-tight:  1.3;
--leading-normal: 1.5;
--leading-loose:  1.7;

--tracking-tight: -0.02em;
--tracking-wide:   0.06em;
```

### Typography rules

- Max four font sizes per screen
- Uppercase labels: `--text-xs` + `--tracking-wide`
- Monospace: `'JetBrains Mono', 'Fira Code', monospace`

---

## Spacing (4px base)

| Token | px |
|-------|-----|
| 1 | 4 |
| 2 | 8 |
| 3 | 12 |
| 4 | 16 |
| 5 | 20 |
| 6 | 24 |
| 8 | 32 |
| 12 | 48 |
| 16 | 64 |

All padding, margin, and gap must be multiples of 4px.

---

## Radius and depth

```css
--radius-sm:   4px;
--radius-md:   8px;
--radius-lg:   12px;
--radius-xl:   16px;
--radius-full: 9999px;

--shadow-sm:  0 1px 3px rgba(0,0,0,0.4);
--shadow-md:  0 4px 12px rgba(0,0,0,0.5);
--shadow-lg:  0 8px 32px rgba(0,0,0,0.6);

--glow-accent: 0 0 0 2px var(--accent-muted), 0 0 16px rgba(124,106,247,0.15);
```

On dark UIs prefer borders + layered backgrounds over heavy shadows (see anti-patterns).

---

## Components

### Buttons

- **Primary:** `bg` accent, `text` inverse, hover `accent-hover`
- **Secondary:** `bg` elevated, `border` default, hover `bg-overlay`
- **Ghost:** transparent, `text` secondary, hover elevated + primary text
- **Danger:** transparent, `border` error ~20%, `text` error, hover error ~10% bg

Shared: `--radius-md`; heights 32 / 36 / 40 (compact / default / large); min-width 80px; padding e.g. `px-3 py-1.5`. States: `:hover`, `:focus-visible` (glow), `:disabled` (~opacity 40%).

### Inputs

- Background `--bg-input`; border default → focus strong + `--glow-accent`
- Text primary; placeholder muted; label secondary, sm, medium, `mb-1.5`

### Cards / panels

- Background elevated; `1px` border subtle; radius lg; padding 16 (compact) or 24 (default)
- Hover: border → default

### Tables

- Header: surface bg, secondary text, xs uppercase tracking-wide
- Row: bottom border subtle; hover elevated
- Cell: primary sm; `py-3 px-4`
- No striping; use hover

### Badges

- Success / warning / error / info: tinted bg (~`18` hex), matching text, border ~`30`
- Neutral: elevated, secondary text, default border
- Radius full; `px-2 py-0.5`; xs medium

---

## Motion

```css
--duration-fast:   100ms;
--duration-normal: 150ms;
--duration-slow:   250ms;
--ease-default: cubic-bezier(0.16, 1, 0.3, 1);
```

- Transition bg, border-color, color, box-shadow on interactive elements
- Avoid UI chrome animations >300ms
- Prefer skeletons over spinners for content loading
- Entrance: opacity + `translateY(4px)` when appropriate

---

## Accessibility

- Contrast: ≥4.5:1 body, ≥3:1 large text (WCAG AA)
- `:focus-visible` ring via accent glow
- Do not rely on color alone — add icon or text
- `cursor: pointer` on clickable non-`<button>` elements
- Full keyboard support for interactive components

---

## Anti-patterns

- Pure black bg or pure white text
- More than two accent hues fighting on one screen
- Heavy box shadows instead of borders/layers on dark UI
- Decorative gradients (tiny gradient on accent CTA only if needed)
- Font sizes outside the defined scale for that screen
- Off-grid spacing (7px, 11px, 15px, etc.)
- Micro-interactions >300ms
- Placeholder as the only label
- Generic “AI” UI: flat white cards, default blue primary, stock layouts

---

## Stack defaults

When the project has no conflicting convention:

- **CSS:** Tailwind v4 (utilities; `dark:` when needed)
- **Components:** shadcn/ui base, restyled to this system
- **Icons:** Lucide React, stroke 1.5
- **Fonts:** Inter or Geist via `next/font` or CDN
- **Motion:** Framer Motion for complex; CSS transitions for simple

---

## Non-web projects

Map tokens to the host framework (e.g. Streamlit `theme`, matplotlib dark style): same hierarchy (base/surface/elevated), one accent, 4px-ish spacing, no pure black/white.
