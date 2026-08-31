# ClearCam Audit — all surfaces (2026-08-31)

Surfaces: `mainview.html` (sole web surface; the legacy `cameraview.html` standalone page was retired — its route redirects to the main surface). The native SwiftUI shell (window chrome, menu bar, settings pane) is thin and follows system conventions; not scored under this web audit.

## Audit Health Score

| # | Dimension | Score | Key Finding |
|---|-----------|-------|-------------|
| 1 | Accessibility | 3 | "Alerts Only" checkbox is a 13×13px target; alert-table help is hover-only |
| 2 | Performance | 3 | Lazy thumbnails, visibility-gated polling; journal re-renders via full innerHTML |
| 3 | Theming | 2 | Two token systems coexist (legacy `--bg-*` vs observatory `--ink/--paper`), stitched with ~70 `!important` overrides and 25 off-token colors |
| 4 | Responsive Design | 3 | Zero horizontal overflow at 375px; a few 41px-tall buttons under the 44px floor |
| 5 | Implementation Integrity | 3 | Coherent, product-specific, honest-state UI; legacy dead CSS + 102 inline styles remain |
| **Total** | | **14/20** | **Good — address weak dimensions** |

## Implementation Integrity Verdict
**Pass.** The surface expresses a coherent product-specific system (household observatory: ink shell, paper work surfaces, sage/brass/coral semantics, honest operating-state language, real camera imagery). Detector: 71 findings — 69 advisories are radii/colors in the superseded legacy CSS layer (verified mostly overridden but still shipped), and both `broken-image` warnings are verified-intentional runtime-populated elements (`#previewImage`, `#enrollPreview`).

## Findings

**P1 — Dual token system / cascade debt.** Location: mainview.html CSS (legacy layer ~lines 20–1330 vs observatory layer ~1360+). Impact: every change risks specificity wars (the invisible Add Camera button this week was this exact failure); 25 colors and 44 radii live outside DESIGN.md tokens. Recommendation: delete or fold the legacy layer into the observatory tokens, promote the recurring one-offs (`#52635d`, `#edf3ef`, hairline rgba) into named tokens, retire the `!important` stitching. Suggested: `$impeccable extract`.

**P2 — Sub-44px touch targets.** Location: `#alertsOnlyCheckbox` (13×13), `.primary-action`/`#loadMoreEventsBtn`/`.secondary-action` (41px tall). Impact: hard to hit on trackpads/touch; WCAG 2.5.8. Recommendation: 17px checkbox with padded label hit area (pattern already used in the alerts modal), bump button padding to reach ≥44px. Suggested: `$impeccable adapt`.

**P2 — Hover-only help in alert table.** Location: "Send Alerts"/"Zone" `?` tooltips in the Detection & Counts modal. Impact: keyboard and touch users can't reach the explanation. Recommendation: the always-visible caption pattern already adopted in Global Settings. Suggested: `$impeccable clarify`.

**P3 — Inline styles (102).** Maintainability drift, not user-facing. Fold into classes opportunistically during other passes.

**P3 — Journal refresh rebuilds via innerHTML.** Fine at 20/page; revisit only if page size grows.

## Positive Findings
- Zero unlabeled interactive controls; logical H1→H4 hierarchy; main/nav/aside landmarks; dialog focus management; `role=status/alert` live regions; reduced-motion alternatives in 3 blocks; explicit body background (committed single look).
- Zero horizontal overflow at 375px; tables scroll inside containers; 44px rail/camera actions.
- Honest state language throughout (engine states, pending descriptions, storage warnings, "Always" schedules).
- Lazy event thumbnails; polling gated on document visibility; toasts/switches animate on compositor properties.

## Recommended Actions
1. **[P1] `$impeccable extract`** — unify the two token systems; promote recurring hex values to DESIGN.md tokens; delete the legacy layer and its `!important` stitching.
2. **[P2] `$impeccable adapt`** — touch-target floor: alerts-only checkbox and 41px buttons.
3. **[P2] `$impeccable clarify`** — replace the alert table's hover-only `?` tooltips with visible captions.
4. **[P?] `$impeccable polish`** — final pass after the above.
