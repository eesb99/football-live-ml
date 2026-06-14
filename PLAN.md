# World Cup Prediction Dashboard Plan

## Current Strategy

Build an individual-budget World Cup prediction dashboard on top of the existing API-Football integration. Keep API-Football as the only active provider and avoid paid enterprise providers, deployments, databases, and external mutations.

## Phase Scope

- Add local Elo-style team ratings with fallback rating `1500`.
- Add pre-match prediction for scheduled fixtures.
- Add combined pre-match plus live prediction for in-play fixtures.
- Add model-driver explanations for why probabilities move.
- Add prediction snapshots under `data/predictions/`.
- Add rating snapshots under `data/ratings/`.
- Refactor the Streamlit app into World Cup prediction views while preserving `World Cup live`, `World Cup season`, and `All live`.

## Success Criteria

- `python3 -m pytest tests` passes.
- `python3 -m compileall src app tests` passes.
- Streamlit app execution test passes.
- `http://localhost:8501` is reachable.
- Dashboard shows Match Board, Pre-Match Prediction, Live Prediction, Model Drivers, Backtest, and Snapshots surfaces.

## Next Steps

1. Verify Streamlit on localhost after the prediction dashboard refactor.
2. Use real World Cup season responses to inspect data completeness when API quota allows.
3. Consider adding bounded historical fixture ingestion only if pre-match ratings need more data.

## Implemented This Phase

- Added `src/ratings.py` for local Elo-style team ratings.
- Added `src/predictor.py` for pre-match, live, and final prediction modes.
- Added prediction snapshots under `data/predictions/`.
- Added rating snapshots under `data/ratings/`.
- Refactored Streamlit into Match Board, Pre-Match Prediction, Live Prediction, Model Drivers, Backtest, and Snapshots tabs.
- Added Calendar tab and MYT schedule columns for World Cup fixtures.
- Added unit tests for ratings, prediction outputs, live blending, drivers, and snapshot rows.
