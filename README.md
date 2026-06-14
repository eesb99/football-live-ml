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
- simple backtest table for completed fixtures

The model is intentionally simple and educational. It is not betting advice.

## Requirements

- Python 3.11+
- API-Football / API-Sports v3 API key

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
- **Backtest**: completed-fixture prediction check with cumulative running accuracy for our model and optional API-Football prediction benchmark accuracy.
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
    ratings.py
    storage.py
  app/
    streamlit_app.py
  data/
    predictions/
    ratings/
    snapshots/
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
data/ratings/
```

Snapshot files contain fixture metadata, model outputs, capture timestamps, and row counts. Empty-live-match refreshes are logged too, so refresh history remains auditable even when no fixtures are live.

- `data/snapshots/`: raw live/fixture refresh snapshots.
- `data/predictions/`: prediction outputs with model version and prediction mode.
- `data/ratings/`: local Elo/team-rating snapshots.

## Model And Features

The current prediction stack is a v2 rules-based engine. It deliberately does not train a model yet.

Training should wait until there are enough stored snapshots paired with final match outcomes for backtesting and calibration.

The prediction stack has two layers:

- **Pre-match**: local Elo-style team ratings with fallback rating `1500`, home advantage, expected goals, and home/draw/away probability.
- **Live**: pre-match probability as the prior, then transparent Poisson live-state update using API-Football statistics/events.

The combined predictor blends the pre-match prior with live match state when a fixture is live.

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

- The API key is read only from `API_FOOTBALL_KEY`.
- The API request header is `x-apisports-key`.
- Missing API keys, API errors, empty live-match responses, and quota/rate-limit responses are handled with explicit exceptions and dashboard messages.
- The active live fixture provider is API-Football only. Optional paid-data adapter interfaces exist, but Opta, Sportradar, SportMonks, databases, deployments, and real paid-provider implementations are intentionally out of scope for the current individual-budget version.
