---
name: ClearCam
description: A calm, local household observatory for one owner.
colors:
  ink: "#101a1c"
  ink-raised: "#172527"
  ink-soft: "#203235"
  paper: "#e8eeeb"
  paper-muted: "#c4d0ca"
  sage: "#99b7a8"
  sage-strong: "#c2decf"
  brass: "#e7b35d"
  danger: "#e58a84"
  line-faint: "rgba(232,238,235,0.12)"
  line-strong: "rgba(232,238,235,0.28)"
  paper-raised: "#f8fbf9"
  paper-tint: "#edf3ef"
  ink-on-paper: "#31413b"
  muted-on-paper: "#52635d"
  border-on-paper: "#b4c0ba"
typography:
  display:
    fontFamily: "-apple-system, BlinkMacSystemFont, SF Pro Text, Helvetica Neue, sans-serif"
    fontSize: "1.05rem"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "-0.01em"
  body:
    fontFamily: "-apple-system, BlinkMacSystemFont, SF Pro Text, Helvetica Neue, sans-serif"
rounded:
  control: "10px"
  chip: "12px"
  panel: "16px"
  pill: "999px"
spacing:
  control-gap: "4px"
  reel-gap: "14px"
  section-gap: "34px"
components:
  button-primary:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    padding: "11px 15px"
  camera-action:
    backgroundColor: "transparent"
    textColor: "{colors.paper-muted}"
    hoverBackgroundColor: "{colors.ink-soft}"
    width: "40px"
    height: "40px"
    rounded: "9px"
---

# Design System: ClearCam

Refined on 6 September 2026 ("refined observatory", approved direction). The identity is unchanged; the chrome got quieter and denser so footage and the journal carry the page.

## Overview

**Creative North Star: "Household observatory"**

A private, local record of what the home has noticed. The interface is calm, capable and plain-spoken. Real camera footage carries the visual weight; controls help the owner watch, review and configure without turning the home into an enterprise surveillance dashboard.

**Key Characteristics:**

- Video leads; the journal supports review.
- Ink surroundings and mineral-paper working surfaces.
- Restrained sage interactions and explicit operating-state language.
- Native-feeling system typography, generous hit areas and quiet SVG icons.

## Colors

Deep ink contains the interface; mineral paper gives review work a readable surface. Sage is reserved for action and selection, brass for focus, and muted coral for destructive action.

### Primary

- **Mineral paper:** the primary action and journal surface, with ink text.
- **Pale sage:** primary-action hover and selected-state emphasis.

### Neutral

- **Deep ink:** page canvas.
- **Raised ink:** camera and dialog surfaces.
- **Soft ink:** camera-action backgrounds and interaction states.
- **Quiet paper:** supporting copy on dark surfaces.

**The Semantic Pair Rule.** An icon's foreground and background form one component state. Never combine the coral delete icon with the old saturated red button background. The corrected default pair measures approximately 5:1 contrast.

## Typography

System sans preserves the approved macOS familiarity. There is no display heading any more: the product name sits at 1.05rem on one topbar line with its tagline, and section titles are small uppercase labels in dim paper (`.8rem`, `+0.02em`) so the content, not the chrome, is the largest thing on screen. Tabular numerals are set on `body`.

**The Readable Form Rule.** Modal and mobile filter text stays at 16px. Adapt the layout rather than shrinking the complete dialog.

## Layout

The wide shell uses a 72px translucent navigation rail (blurred ink, hairline seam) and content capped at 1360px. A single-line topbar (name, tagline, engine state pill, one primary action) precedes the camera grid; the journal and system ledger follow in a 1fr / 296px grid that stacks under 1080px. DOM order agrees with visual and keyboard order: journal first, ledger second.

The wide reel scrolls horizontally. Ordinary cards use `clamp(280px, 32vw, 430px)`; the first or selected card expands to `clamp(540px, 62vw, 820px)`. Video has a 16:9 well. Three camera actions share an aligned row below it.

At 980px and below, navigation becomes a bottom dock and the journal/ledger stack. At 650px and below, camera cards stack at full available width, with three equally sized action buttons. The action hit area is at least 44px high across the tested desktop, tablet and phone widths.

The journal's filter row, results and pagination are separate groups. Event cards use one column with thumbnail, metadata and actions. On narrow screens the actions move beneath the thumbnail and metadata.

## Elevation & Depth

Elevation is declared once per surface: camera cards and dialogs use an inset hairline (no outer border) plus, when selected or floating, one soft downward shadow. The paper journal floats on a single deep soft shadow. Dialog scrims and the rail use backdrop blur as a material, and fall back to solid ink under `prefers-reduced-transparency`. A global `[hidden]` rule outranks every component display rule.

**The State Before Decoration Rule.** Health comes from the engine endpoint, not a decorative green dot. A live connection failure cannot be represented as successful monitoring just because old footage still plays.

## Shapes

Panels use softly rounded corners; controls are smaller and more compact. Icon buttons keep a rectangular hit area even when their mark is small. Status treatments may be rounded, but long status text must wrap.

## Components

### Buttons

Primary actions reverse to paper and ink, with sage hover. Camera actions use soft ink and paper; deletion uses the contrasting coral foreground. All actions retain visible focus, accessible names and disabled/pending states.

### Inputs / Fields

Every form field has a persistent label. Inputs preserve unsaved edits through errors and background refreshes. Saving stays pending until the server confirms success. Connection credentials remain masked in the add-camera field.

### Navigation

The dark rail uses embedded SVG. The search action focuses usable search or explains how to enable it while leaving camera/date filters available. The bottom dock is the responsive form of the same navigation.

### Camera reel

Camera names are plain text, not HTML or JavaScript. Action handlers bind to the underlying name safely. Video playback uses a bundled, pinned HLS library so the control surface does not depend on a CDN.

### Daily journal

Event thumbnails are real local media and load lazily. Generated descriptions are labeled “AI description” and rendered as text. Empty results are distinct from a failed fetch. Preview and playback actions work with keyboard input.

### System ledger and dialogs

The ledger shows engine, detection, active notification-rule count, AI state and selected review date without claiming that OS notification permission was verified. Dialogs carry names and modal semantics, move focus inside, contain keyboard traversal and restore focus on close. Escape closes the current dialog. Smaller screens reflow contents without CSS zoom.

**The Honest Waiting Rule.** “Waiting for an event” does not claim that an AI description has been generated. Recording, detection, event persistence and description generation are distinct steps.

## Do's and Don'ts

### Do:

- **Do** retain the approved local-first product language and real camera imagery.
- **Do** keep all three camera actions aligned with at least 44px hit areas.
- **Do** preserve edits and explain recovery when a request fails.
- **Do** show actual engine state and label generated descriptions.
- **Do** use reduced-motion alternatives without changing the final state.

### Don't:

- **Don't** replace the household observatory with a new visual identity during polish.
- **Don't** hide core controls to make a narrow layout fit.
- **Don't** represent stale playback as a healthy live camera connection.
- **Don't** interpolate camera names into HTML or inline JavaScript.
- **Don't** claim completed native macOS packaging or a verified notification delivery path from these web-surface checks.
