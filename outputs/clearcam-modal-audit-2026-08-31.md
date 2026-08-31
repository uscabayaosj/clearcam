# ClearCam audit — camera-card modals (2026-08-31)

Scope: the four dialogs reachable from a camera card — Edit Zone (`#zoneModal`, gear), Detection & Counts (`#alertsModal`, bell), Add Alert (`#alertModal`), and the delete confirmation (`#confirmModal`). "Settings modal" resolves to Edit Zone; Global Settings is not card-stemming and was audited previously.

## Audit Health Score (modal set)

| # | Dimension | Score | Key Finding |
|---|-----------|-------|-------------|
| 1 | Accessibility | 3 | Zone hint text ~3.6:1; zone drawing is mouse-only |
| 2 | Performance | 4 | — |
| 3 | Theming | 2 | Edit Zone is pre-token: #ccc/#666 literals, inline styles, legacy button cascade |
| 4 | Responsive | 3 | All four fit 375px with no overflow; zone action row survives by shrinking |
| 5 | Implementation Integrity | 3 | Three modals fully in-system; Edit Zone is the unconverted legacy screen |
| **Total** | | **15/20** | Good — one modal drags the set |

## Verdict
Three of the four modals are coherent with the household observatory (Detection & Counts and Add Alert were rebuilt this session; the confirm dialog was born in-system and inherits dialog semantics automatically). **Edit Zone is the fossil**: it predates every design pass and reads like a different product.

All four verified with `role=dialog`, `aria-modal`, `aria-labelledby`, Escape close, and 44px buttons (the global touch-floor rule reached even the legacy modal).

## Findings

**P1 — Edit Zone shows the camera credential in plain text.** `#zoneSource` ("Change source") is `type=text`; typing a credentialed rtsp:// URL renders the password on screen. Add Camera correctly uses `type=password`. Impact: shoulder-surfing + inconsistent security posture. Fix: password type + the same placeholder/hint pattern as Add Camera. → `$impeccable harden`

**P1 — Edit Zone's button hierarchy is inverted.** Clear Zone (destructive) is the loudest element (solid red, largest), while Cancel and Save are visually identical pale-sage twins. Impact: the dangerous action attracts the eye; the primary action is findable only by position. Fix: Save gets primary treatment, Cancel quiet, Clear Zone the coral-quiet destructive pattern used elsewhere. → `$impeccable polish`

**P2 — Zone hint fails contrast.** "Click on the image to make a detection zone." is `#666` on ink-raised ≈ 3.6:1 (< 4.5:1). Fix: `--paper-muted`. → `$impeccable polish`

**P2 — Zone drawing has no keyboard path.** The polygon is click-only with no alternative; at minimum the modal should state the limitation and keep threshold/outside/source keyboard-operable (they are). → `$impeccable harden`

**P2 — Save gives no feedback and the title names no camera.** `saveZone()` closes silently; "Edit Zone" lacks the `· camera` subject every other modal now carries. → `$impeccable polish`

**P3** — Alert-builder chips are 36px tall (pointer-fine, sub-44 for touch); zone checkbox uses default accent; Add Alert's schedule section still center-aligned from the old layout.

## Positive Findings
- Dialog semantics are uniform across all four — including the brand-new confirm dialog, which the generic modal wiring picked up automatically.
- Zero overflow at 375px for the whole set; the 44px floor held everywhere.
- Detection & Counts and Add Alert now carry the strongest patterns in the product (visible captions, live rule sentence, honest schedules).

## Recommended Actions
1. **[P1] `$impeccable harden`** — Edit Zone: mask the source credential, state the keyboard limitation.
2. **[P1] `$impeccable polish`** — Edit Zone: tokens, button hierarchy, hint contrast, camera subject, save feedback.
