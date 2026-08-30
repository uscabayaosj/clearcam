# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

The first release serves one person managing cameras in their own home. They use a Mac that remains on as their private camera hub.

## Product Purpose

ClearCam turns a user's Mac into a private AI security hub for their existing cameras. It records locally, detects useful events, and helps the owner understand what happened without requiring continuous cloud video processing.

## Positioning

ClearCam keeps camera ingest, AI detection, recordings, and event search on the owner's Mac by default. It is a personal, local-first camera hub rather than a cloud NVR subscription.

## Operating Context

The owner opens ClearCam at home to see whether cameras are healthy, review recent events, search their footage, configure detection zones and alerts, and manage local storage. The product should feel like a native macOS control surface while the existing Python engine runs locally. The initial release is intentionally local: remote viewing is out of scope, while local notifications remain in scope.

## Capabilities and Constraints

- Existing camera ingest, recording, object detection, tracking, event search, zones, alerts, footage analysis, and settings remain functional.
- The local Python engine requires persistent runtime, local disk storage, FFmpeg, model weights, and access to home-network cameras.
- The first release supports one owner only; multi-user roles, shared households, and remote viewing are not in scope.
- The UI is currently served as local HTML by the Python engine and will be reshaped into a macOS-app-like control surface.
- Security hardening is required before any public network exposure.

## Brand Commitments

ClearCam is private, calm, capable, and plain-spoken. It must avoid implying that video leaves the owner's Mac by default.

## Evidence on Hand

- Existing product code and local UI: `clearcam.py`, `mainview.html`, `cameraview.html`.
- Product logo: `images/logo.png`.
- Existing documentation demonstrates camera feeds, detection, notifications, event clips, and local installation in `README.md`.
- No approved customer testimonials, pricing claims, performance benchmarks, or cloud-security certifications are on hand.

## Product Principles

- Local video by default.
- Make camera health and recent activity immediately legible.
- Surface the next useful action without turning routine monitoring into work.
- Keep advanced detection controls available but out of the daily path.
- Earn trust with clear system state rather than surveillance theatrics.

## Accessibility & Inclusion

The local control surface must support keyboard operation, visible focus, sufficient contrast, descriptive labels, and reduced-motion preferences.
