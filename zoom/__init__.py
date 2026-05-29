"""Stateful 3-handed zoom/fast-fold advisory service (layers L2–L5).

A SEPARATE package from the frozen, competition-critical `pokerbot` package.
It runs alongside the frozen brain, reuses its WebSocket protocol, and imports
ONLY pure abstraction functions from `pokerbot` (read-only — never modified).

Layers:
  L2  opponent_model  — per-opponent Dirichlet action model (built + tested)
  L3  range_tracker   — Bayesian hole-card range filter (built + tested)
  L4  subgame solver  — NOT built (future milestone)
  L5  serve_zoom      — advisory service skeleton (scripts/serve_zoom.py)

Integration seams to the real repo live in `abstraction_bridge` and
`archetype_bridge`; neither L2 nor L3 imports `pokerbot` directly.
"""
