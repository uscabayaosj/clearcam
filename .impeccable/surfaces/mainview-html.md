---
version: 1
slug: "mainview-html"
primary_target: "mainview.html"
related_targets: []
---

# Main Control Surface

## Scope and mode

`mainview.html` is the local ClearCam control surface. Mode: Operate.

## Audience, task, and constraints

One owner uses an always-on Mac at home to check cameras, review events, search footage, and configure local detection. Keep all existing camera, alert, search, storage, zone, footage-analysis, and settings functions available. No remote-viewing claims.

## Chosen direction

Household observatory, using the approved [camera reel + daily journal comp](.impeccable/mocks/clearcam-observatory-b.png). A reel makes camera switching immediate; the lower journal makes a day of observation easy to understand; the health ledger keeps system trust visible without behaving like an enterprise dashboard.

## Component grammar and inventory

| Ingredient | Implementation medium | Commitment |
| --- | --- | --- |
| Application rail and navigation marks | Semantic HTML and inline SVG | A slim dark rail; icon buttons retain accessible labels. |
| Camera reel | Existing dynamic video elements plus HTML/CSS | Variable-width horizontal rail; selected stream is visually largest. |
| Daily journal | Existing dynamic event-image markup plus HTML/CSS | Dense but calm event rows, with one primary focal event in actual content. |
| Health ledger | Semantic HTML/CSS backed by existing camera settings and controls | A quiet, readable system-status area rather than a metric wall. |
| Add camera action | Existing form and semantic HTML/CSS | One high-visibility action at the header; no decorative treatment. |
| Camera imagery | Existing live video and locally produced previews | No invented surveillance imagery. |

## Type, surface, and motion

Dark ink background; mineral-paper content surfaces; hairline seams and softly rounded 14px panels; no gradients or glass. System sans type retains macOS familiarity, with tabular numerals for timestamps. The camera reel changes selection with one deliberate crossfade/edge slide; reduced-motion users see the final state directly.

## Unresolved decisions

The first native macOS wrapper and its menu-bar interaction remain a later implementation step; this surface is the local HTML control plane it will host.
