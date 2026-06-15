# football-live-ml

World Cup football prediction dashboard using API-Football / API-Sports v3 data.

The dashboard supports individual-budget World Cup prediction workflows:

- home win probability
- draw probability
- away win probability
- next goal probability
- home scores next probability
- away scores next probability
- pre-match prediction from local Elo-style team ratings
- live prediction from match events and statistics
- model-driver explanations
- model source/status display
- API-Football prediction endpoint comparison
- proxy xG fallback when real xG is unavailable
- expected remaining goals
- model confidence
- local prediction snapshots
- fair walk-forward backtest table for completed fixtures
- benchmark miss and API-disagreement diagnostics
- draw-specific risk labels and miss diagnostics
- SportMonks World Cup All-in access audit, sanitized cache, provider-status dashboard, and guarded candidate benchmark

The model is intentionally simple and educational. It is not betting advice.

## Requirements

- Python 3.11+
- API-Football / API-Sports v3 API key
- optional SportMonks API token for World Cup All-in access auditing and cache refreshes

## Setup

1. Create and activate a virtual environment:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a local environment file:

   ```bash
   cp .env.example .env
   ```

4. Edit `.env` and set:

   ```text
   API_FOOTBALL_KEY=your_api_key_here
   SPORTMONKS_API_TOKEN=your_sportmonks_api_token_here
   ```

## Run The Dashboard

```bash
streamlit run app/streamlit_app.py
```

The app fetches all live fixtures from:

```text
https://v3.football.api-sports.io/fixtures?live=all
```

For each selected or displayed live match, it can also fetch:

- `/fixtures/events?fixture={fixture_id}`
- `/fixtures/statistics?fixture={fixture_id}`
- `/predictions?fixture={fixture_id}` for optional benchmark comparison

## World Cup Focus

The dashboard defaults to **World Cup live** mode.

Use the sidebar to switch between:

- **World Cup live**: fetches all live fixtures, filters to World Cup fixtures, then models each live World Cup match.
- **World Cup season**: lists World Cup fixtures for the selected league ID and season. The default league ID is `1` and the default season is `2026`. Scheduled fixtures use pre-match prediction; live fixtures use pre-match plus live-state updates.
- **All live**: shows all live fixtures across competitions.

The sidebar also lets you adjust the World Cup league ID and season if your API-Football plan/account uses a different competition mapping.

If API-Football rejects the selected season on the current plan, the app automatically tries accessible fallback seasons `2022`, `2023`, and `2024`, then shows which season is actually displayed.

Dashboard tabs:

- **Schedule**: calendar and match board grouped by Malaysia Time (MYT, UTC+8).
- **Predictions**: home/draw/away probabilities, next-goal outputs, outcome bars, and a win/loss/pending result banner when an actual final result is available.
- **Model Breakdown**: model mode, data source availability, Elo prior, extracted features, proxy/real xG status, live strength components, Poisson expected goals, and explanation drivers.
- **Model Comparison**: side-by-side benchmark view comparing `Our v2 rules model` with `API-Football prediction` when the API-Football predictions endpoint is available.
- **Backtest**: fair walk-forward completed-fixture benchmark with cumulative accuracy, shared-fixture API-Football comparison, Brier score, log loss, unavailable benchmark counts, confidence diagnostics, miss/API-disagreement tables, draw-specific diagnostics, a non-leaky team-prior ablation, and a guarded SportMonks candidate experiment.
- **Paper Trading**: research-only World Cup lifecycle paper ledger using cached SportMonks odds, capped fractional Kelly sizing, open/settled status, realized paper P&L, open exposure, odds movement, CLV direction, edge drift, and first-entry versus latest-entry paper P&L. Real stake remains zero while the benchmark gate is blocked.
- **Provider Status**: local SportMonks token/audit status, latest sanitized audit path, generated cache status, API-Football to SportMonks mapping coverage, subscription metadata, and rate-limit metadata.
- **Snapshots**: latest local prediction snapshot files.

## Project Structure

```text
football-live-ml/
  README.md
  requirements.txt
  .env.example
  .gitignore
  src/
    config.py
    adapters.py
    api_client.py
    external_predictions.py
    features.py
    model.py
    predictor.py
    paper_trading.py
    ratings.py
    storage.py
    team_priors.py
    sportmonks_audit.py
    sportmonks_client.py
    sportmonks_enrichment.py
    sportmonks_mapping.py
    benchmark.py
  app/
    streamlit_app.py
  data/
    api_predictions/
    predictions/
    ratings/
    snapshots/
    team_priors/
    sportmonks/
  tests/
    test_features.py
    test_predictor.py
    test_adapters.py
```

## Snapshots

Every dashboard refresh writes a CSV snapshot to:

```text
data/snapshots/
data/predictions/
data/api_predictions/
data/ratings/
data/team_priors/
data/sportmonks/
```

Snapshot files contain fixture metadata, model outputs, capture timestamps, and row counts. Empty-live-match refreshes are logged too, so refresh history remains auditable even when no fixtures are live.

- `data/snapshots/`: raw live/fixture refresh snapshots.
- `data/predictions/`: prediction outputs with model version and prediction mode.
- `data/api_predictions/`: normalized API-Football `/predictions` cache files keyed by fixture ID. These are generated local cache files and are ignored by Git.
- `data/ratings/`: local Elo/team-rating snapshots.
- `data/team_priors/`: optional curated pre-match team-prior source files. The app expects a real `team_priors.csv`; it does not fabricate production priors.
- `data/sportmonks/`: generated SportMonks audit and provider-cache workspace. JSON files under this tree are ignored by Git; `.gitkeep` files preserve the directory layout.

## SportMonks Access Audit

SportMonks is wired as an audited and cached provider. It is not promoted into the headline scoring model unless the separate candidate benchmark proves improvement without data leakage.

Set `SPORTMONKS_API_TOKEN` in `.env`, then run:

```bash
python3 -m src.sportmonks_audit
```

The audit writes sanitized JSON to:

```text
data/sportmonks/audits/
```

The audit checks league discovery, World Cup 2026 season discovery, fixtures for SportMonks season `26618`, fixture detail, expected goals, news, SportMonks probabilities, pre-match odds, match facts, subscription metadata, and rate-limit metadata. It never persists the raw token or raw token-bearing URLs.

Latest verified live audit baseline:

- World Cup league ID discovered: `732`
- World Cup 2026 season ID discovered: `26618`
- selected fixture ID: `19606945`
- accessible categories: leagues, World Cup search, seasons, World Cup 2026 fixtures, fixture detail, expected goals, news
- empty but valid categories on the selected future fixture: SportMonks predictions, pre-match odds, match facts
- error categories: none

## SportMonks Cache And Candidate Benchmark

Refresh the local SportMonks cache explicitly:

```bash
python3 -m src.sportmonks_enrichment
```

The cache command writes sanitized JSON under:

```text
data/sportmonks/fixtures/
data/sportmonks/odds/
data/sportmonks/xg/
data/sportmonks/news/
```

It fetches SportMonks World Cup 2026 fixtures, fixture detail for the first configured fixture window, expected-goals records, and pre-match news records. Generated JSON files are ignored by Git and should not contain `SPORTMONKS_API_TOKEN` or raw `api_token=` values.

The dashboard uses these files only as local cache input:

- **Provider Status** maps API-Football fixtures to SportMonks fixtures by kickoff and team names, then shows detail/xG/news coverage.
- **Backtest** builds a SportMonks candidate lane from mapped enrichment.
- SportMonks xG is treated as `post_match_only` unless metadata proves it was available before kickoff, so it is blocked from pre-match walk-forward scoring by default.
- The headline model remains `world-cup-rules-v2` unless the candidate beats the current model on both Brier score and log loss with enough eligible non-leaky fixtures.

## Market Odds And CLV Gate

Capture pre-kickoff SportMonks odds snapshots explicitly:

```bash
python3 -m src.market_intelligence capture
```

The market layer only normalizes full-time result / match-winner odds. It does not use player props, corners, totals, handicaps, enhanced prices, or live markets.

The market logic:

- groups full-time result odds by bookmaker
- requires a complete home/draw/away triplet
- converts decimal odds to implied probabilities
- removes bookmaker overround per bookmaker
- averages no-vig probabilities into a market consensus
- tracks the best available decimal price per outcome
- compares our model probability against market-implied probability
- computes expected value at the best available price
- tracks closing-line value when two or more pre-kickoff snapshots exist

Backtest gating is deliberately strict. A row can only become a paper-trade candidate when:

- the market snapshot was captured before kickoff
- model edge clears the configured threshold
- expected value clears the configured threshold
- the benchmark gate passes through Brier score and log-loss evidence

If the benchmark gate is blocked, market rows remain research diagnostics. The dashboard does not promote them into betting advice.

## Model And Features

The current prediction stack is a v2 rules-based engine. It deliberately does not train a model yet.

Training should wait until there are enough stored snapshots paired with final match outcomes for backtesting and calibration.

The prediction stack has two layers:

- **Pre-match**: local Elo-style team ratings with fallback rating `1500`, home advantage, expected goals, and home/draw/away probability.
- **Live**: pre-match probability as the prior, then transparent Poisson live-state update using API-Football statistics/events.

The combined predictor blends the pre-match prior with live match state when a fixture is live.

Pre-match confidence is based on the probability distribution shape, margin between the top two outcomes, rating sample depth, fallback-rating status, and probability uncertainty. Close fixtures with shallow rating history are intentionally low-confidence.

The pre-match layer includes a conservative close-match draw calibration: when expected goals are close and total expected goals are low/moderate, draw probability is lightly lifted while preserving normalized home/draw/away probabilities.

Draw calibration also considers rating gap, recent-form gap, and top-vs-draw margin so draw probability is only lifted when multiple close-match signals agree. Benchmark rows expose draw risk labels, expected-goal gap, total expected goals, rating/form gaps, draw rank, and top-vs-draw margin so draw misses can be audited directly.

Feature extraction includes:

- match state: minute, elapsed fraction, remaining fraction, score difference
- discipline: red cards, yellow cards, card differences
- attacking pressure: shots, shots on target, corners, possession, pressure share
- xG quality: real API-Football xG when returned, otherwise proxy xG from shots, shots on target, shots inside box, corners, possession, recent events, and recent goals
- passing and defensive context: pass accuracy, goalkeeper saves, fouls, offsides
- event context: recent events, recent goals, penalty goals
- quality flags: data completeness score
- schedule fields: API fixture date converted to Malaysia Time for calendar display

The Poisson model adjusts expected remaining goals using:

- team pressure share and shot quality
- effective xG pace, using real xG first and proxy xG as fallback
- home advantage
- score-state incentives
- red/yellow card effects
- match tempo
- remaining match time

Optional paid-data adapter interfaces live in `src/adapters.py`:

- odds
- real xG
- injuries
- news

These interfaces are fallback-safe. With no paid provider configured, the dashboard still works and clearly shows missing provider status. If real xG or odds are unavailable, the engine uses proxy xG and Elo/team-rating priors instead.

`src/model.py` still includes a placeholder adapter for a future scikit-learn model.

`src/predictor.py` is the main dashboard-facing prediction API. It returns prediction mode, probabilities, confidence, expected goals, and readable model drivers.

`src/external_predictions.py` normalizes optional API-Football prediction endpoint payloads for comparison only. These predictions are not used in the local `world-cup-rules-v2` calculation.

`src/sportmonks_client.py`, `src/sportmonks_audit.py`, `src/sportmonks_mapping.py`, and `src/sportmonks_enrichment.py` provide the SportMonks foundation: token-safe request handling, sanitized access audits, fixture/news/xG cache refreshes, and fixture mapping by kickoff/team names. SportMonks data is evaluated as a separate candidate lane, not blended into headline prediction outputs.

`src/market_intelligence.py` provides the odds and market-evaluation layer: sanitized pre-kickoff odds snapshots, full-time-result market normalization, no-vig market-implied probabilities, expected-value comparison, CLV tracking, and benchmark-gated paper-trade flags.

`src/paper_trading.py` provides the research-only paper ledger. It sizes paper entries with capped fractional Kelly, settles completed fixtures against actual outcomes, keeps real stake at zero, and reports realized paper P&L plus open exposure. It also compares first-entry versus latest-entry paper P&L when multiple odds snapshots exist. It does not fetch odds directly; odds come from cached SportMonks pre-kickoff snapshots.

`src/market_intelligence.py` also emits odds-movement fields for paper outcomes: first/latest/best/worst seen decimal odds, first/latest market probability, first/latest edge and expected value, edge change, expected-value change, CLV direction, and hours from each entry snapshot to kickoff.

`src/benchmark.py` provides the fair benchmark path. It scores each completed fixture before updating Elo ratings from that fixture result, then compares against API-Football only on shared fixtures where API-Football has usable home/draw/away probabilities. It also has a SportMonks candidate scorer that refuses enrichment unless the mapped provider data is available before kickoff.

During walk-forward benchmarking, `src/benchmark.py` also maintains simple recent-form state from prior completed fixtures only. Form state tracks points, goals for, goals against, and goal difference over a short window, then lightly adjusts later pre-match expected goals without seeing the fixture being scored.

`src/team_priors.py` defines the optional pre-match team-prior schema. Priors must include `team_id`, `team_name`, `strength_rating`, `source`, `source_category`, `as_of`, and `available_before_kickoff`. The non-leak guard rejects late rows and blocked categories such as fixture results, same-tournament results, odds, and provider predictions. When no real `data/team_priors/team_priors.csv` is loaded, the Backtest tab shows the schema and leaves the headline model unchanged.

## Testing

Run:

```bash
python3 -m pytest tests
python3 -m compileall src app tests
python3 -c 'from streamlit.testing.v1 import AppTest; app = AppTest.from_file("app/streamlit_app.py"); app.run(timeout=45); assert not app.exception, [e.value for e in app.exception]; print("streamlit_app_executed")'
```

If `pytest` is not installed in your active environment, install it separately:

```bash
pip install pytest
```

## Notes

- The API-Football API key is read only from `API_FOOTBALL_KEY`.
- The SportMonks API token is read only from `SPORTMONKS_API_TOKEN` and must not be printed, committed, or persisted in generated audit files.
- The API request header is `x-apisports-key`.
- Missing API keys, API errors, empty live-match responses, and quota/rate-limit responses are handled with explicit exceptions and dashboard messages.
- The active scoring and live fixture provider remains API-Football. SportMonks is currently an audit/cache/candidate-benchmark and market-evaluation provider only; databases, deployments, automated staking, and additional paid-provider integrations remain out of scope until explicitly added.
