# World Cup Prediction Dashboard Plan

## Current Strategy

Build an individual-budget World Cup prediction dashboard on top of the existing API-Football integration. Keep API-Football as the active live fixture provider while using SportMonks World Cup All-in as an audited/cacheable enrichment source and separate candidate benchmark. Avoid deployments, databases, and external mutations.

Current calibration posture: neutral World Cup fixtures remove the standard home-advantage Elo boost, host-country World Cup fixtures keep host advantage, and close cold-start neutral fixtures receive material but not forced-top draw-risk calibration. A non-leaky team-prior interface now exists, but no real production `data/team_priors/team_priors.csv` source is loaded yet; the ablation therefore reports zero prior signal and keeps the headline model unchanged. This is a bounded calibration/research surface only; API-Football still leads on accuracy and log loss in the 12-match sample, so market-edge paper-trade flags remain gated.

## Phase Scope

- Add local Elo-style team ratings with fallback rating `1500`.
- Add pre-match prediction for scheduled fixtures.
- Add combined pre-match plus live prediction for in-play fixtures.
- Add model-driver explanations for why probabilities move.
- Add prediction snapshots under `data/predictions/`.
- Add rating snapshots under `data/ratings/`.
- Add fair walk-forward benchmark metrics so completed fixtures are scored before their results update ratings.
- Refactor the Streamlit app into World Cup prediction views while preserving `World Cup live`, `World Cup season`, and `All live`.

## Success Criteria

- `python3 -m pytest tests` passes.
- `python3 -m compileall src app tests` passes.
- Streamlit app execution test passes.
- `http://localhost:8501` is reachable.
- Dashboard shows Match Board, Pre-Match Prediction, Live Prediction, Model Drivers, fair Backtest benchmark, and Snapshots surfaces.
- Provider Status shows the latest sanitized SportMonks audit when `SPORTMONKS_API_TOKEN` is configured.
- Provider Status shows local SportMonks fixture/detail/xG/news cache status and API-Football to SportMonks mapping coverage.
- Backtest shows a SportMonks candidate lane that only evaluates non-leaky pre-kickoff enrichment and keeps the headline model unchanged unless Brier score and log loss improve.
- Backtest shows a non-leaky team-prior ablation when a real pre-match prior source file is available, and exposes the required schema instead of fabricating team ratings when no source is loaded.
- Backtest shows market-implied probability, expected value, CLV tracking, and paper-trade candidate status only after the benchmark gate passes.
- Paper Trading shows cached-odds availability, capped fractional-Kelly paper sizing, open/settled lifecycle status, realized paper P&L, open paper exposure, odds movement, CLV direction, edge drift, first-entry versus latest-entry paper P&L, and time-to-kickoff while keeping real stake at zero.

## Next Steps

1. Verify Streamlit on localhost after the prediction dashboard refactor.
2. Use real World Cup season responses to inspect data completeness when API quota allows.
3. Accumulate more completed World Cup fixtures before using the SportMonks candidate result as a promotion signal.
4. Keep accumulating pre-kickoff odds snapshots so CLV can be measured from repeated market captures.
5. Re-evaluate neutral/draw calibration after more completed fixtures; do not promote betting signals until Brier/log-loss beat API-Football with enough rows.
6. Populate `data/team_priors/team_priors.csv` from a real pre-tournament source, then rerun the prior ablation; do not blend API-Football predictions into the headline model.

## Implemented This Phase

- Added `src/ratings.py` for local Elo-style team ratings.
- Added `src/predictor.py` for pre-match, live, and final prediction modes.
- Added prediction snapshots under `data/predictions/`.
- Added rating snapshots under `data/ratings/`.
- Refactored Streamlit into Match Board, Pre-Match Prediction, Live Prediction, Model Drivers, Backtest, and Snapshots tabs.
- Added Calendar tab and MYT schedule columns for World Cup fixtures.
- Added fair walk-forward benchmark scoring, shared-fixture API-Football comparison, Brier score, log loss, and normalized API prediction cache files.
- Added benchmark miss/API-disagreement diagnostics, probability-margin confidence, conservative close-match draw calibration, and walk-forward recent-form state.
- Added draw-specific benchmark diagnostics, draw risk labels, API draw-miss detection, and more explicit close-match draw calibration gates.
- Added SportMonks World Cup All-in config, redacted client, sanitized access audit, generated-data directories, fixture mapping foundation, Provider Status dashboard tab, and tests. Latest live audit found World Cup league `732`, season `26618`, selected fixture `19606945`, accessible fixtures/detail/xG/news, empty predictions/odds/match-facts for the selected future fixture, and no error categories.
- Added sanitized SportMonks fixture/detail/xG/news cache refreshes, API-Football to SportMonks coverage rows, Provider Status cache/mapping display, and a guarded SportMonks candidate benchmark that refuses post-match or late enrichment.
- Added pre-kickoff SportMonks odds snapshots, no-vig market-implied probabilities, expected-value comparison, CLV tracking, and benchmark-gated paper-trade candidate rows.
- Added World Cup host/neutral venue context and stronger close-neutral draw calibration. On the first 8 completed 2026 World Cup fixtures, Brier improved `0.622 -> 0.614`, log loss improved `1.033 -> 1.020`, and draw misses improved `3 -> 1`, while accuracy stayed `4/8` and API-Football remains ahead.
- Repaired cold-start neutral draw calibration after the completed sample expanded to 12 fixtures. The hard `42%` draw floor was replaced with a soft material-draw target that does not automatically make draw the top prediction; current benchmark improved from `5/12` to `7/12`, Brier improved `0.620 -> 0.609`, log loss improved `1.023 -> 1.012`, and API-Football remains ahead on accuracy/log loss at `8/12`, Brier `0.611`, log loss `0.970`.
- Added `src/team_priors.py`, an optional non-leaky pre-match team-prior schema/loader, conservative capped prior rating adjustments in `src/predictor.py`, and baseline-vs-priors benchmark ablation metrics in `src/benchmark.py` and the Backtest tab. Refreshed benchmark on 72 fetched fixtures / 12 completed shared API rows loaded `0` production priors, so prior rows had `0` signal, `0` changed picks, and matched the baseline: our model `7/12`, Brier `0.609`, log loss `1.012`; API-Football remains `8/12`, Brier `0.611`, log loss `0.970`.
- Added `src/paper_trading.py` and a Paper Trading tab for World Cup lifecycle research. The page reads cached SportMonks odds, applies capped fractional Kelly paper sizing, tracks settled/open paper rows, realized paper P&L, and open exposure, and leaves real stake at `0` while the benchmark gate is blocked.
- Added Odds Movement fields to Paper Trading. The page now tracks first/latest/best/worst pre-kickoff odds for the selected paper outcome, CLV direction, first/latest edge and EV, edge/EV change, first-entry versus latest-entry paper P&L, and hours to kickoff for first/latest snapshots.
- Latest evidence after Odds Movement update: `python3 -m pytest tests/test_paper_trading.py tests/test_market_intelligence.py tests/test_app_helpers.py` passed with 37 tests; `python3 -m pytest tests` passed with 117 tests; `python3 -m compileall src app tests` passed; Streamlit AppTest printed `streamlit_app_executed`; localhost returned `200 text/html`.
- Added unit tests for ratings, prediction outputs, live blending, drivers, and snapshot rows.
- Latest evidence after model-improvement diagnostics update: `python3 -m pytest tests` passed with 60 tests; compileall and Streamlit AppTest passed.
- Latest evidence after draw diagnostics update: `python3 -m pytest tests` passed with 64 tests; compileall and Streamlit AppTest passed.
- Latest evidence after cold-start neutral draw repair: `python3 -m pytest tests` passed with 105 tests; `python3 -m compileall src app tests` passed; Streamlit AppTest printed `streamlit_app_executed`; benchmark gate remains blocked with reason `brier_log_loss_not_better`.
- Latest evidence after team-prior implementation: `python3 -m pytest tests/test_team_priors.py tests/test_predictor.py tests/test_benchmark.py tests/test_app_helpers.py` passed with 58 tests; refreshed benchmark command completed against 72 fixtures and 12 completed/shared API rows; `python3 -m pytest tests` passed with 112 tests; `python3 -m compileall src app tests` passed; Streamlit AppTest printed `streamlit_app_executed`.
