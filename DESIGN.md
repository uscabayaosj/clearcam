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
    fontSize: "clamp(1.8rem, 2.7vw, 2.7rem)"
    fontWeight: 650
    lineHeight: 1.1
    letterSpacing: "-0.03em"
  body:
    fontFamily: "-apple-system, BlinkMacSystemFont, SF Pro Text, Helvetica Neue, sans-serif"
rounded:
  control: "9px"
  panel: "15px"
spacing:
  control-gap: "6px"
  reel-gap: "18px"
  section-gap: "42px"
components:
  button-primary:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    padding: "11px 15px"
  camera-action:
    backgroundColor: "{colors.ink-soft}"
    textColor: "{colors.paper}"
    width: "44px"
    height: "44px"
    rounded: "{rounded.control}"
---

# Design System: ClearCam

Refreshed with approval on 30 August 2026. This records the corrected implementation while preserving the approved household-observatory direction.

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

System sans preserves the approved macOS familiarity. Display headings use the frontmatter's compact scale; section headings are medium weight, while supporting text remains subordinate without uppercase decoration.

**The Readable Form Rule.** Modal and mobile filter text stays at 16px. Adapt the layout rather than shrinking the complete dialog.

## Layout

The wide shell uses a 76px navigation rail and content capped at 1320px. The header precedes the camera reel; the journal and system ledger follow. DOM order agrees with visual and keyboard order: journal first, ledger second.

The wide reel scrolls horizontally. Ordinary cards use `clamp(280px, 32vw, 430px)`; the first or selected card expands to `clamp(540px, 62vw, 820px)`. Video has a 16:9 well. Three camera actions share an aligned row below it.

At 980px and below, navigation becomes a bottom dock and the journal/ledger stack. At 650px and below, camera cards stack at full available width, with three equally sized action buttons. The action hit area is at least 44px high across the tested desktop, tablet and phone widths.

The journal's filter row, results and pagination are separate groups. Event cards use one column with thumbnail, metadata and actions. On narrow screens the actions move beneath the thumbnail and metadata.

## Elevation & Depth

Tonal separation establishes the shell. The selected feed and dialogs use a restrained downward shadow. Borders establish panel edges; navigation and controls stay quiet at rest.

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
