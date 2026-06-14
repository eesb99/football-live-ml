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

## World Cup Focus

The dashboard defaults to **World Cup live** mode.

Use the sidebar to switch between:

- **World Cup live**: fetches all live fixtures, filters to World Cup fixtures, then models each live World Cup match.
- **World Cup season**: lists World Cup fixtures for the selected league ID and season. The default league ID is `1` and the default season is `2026`. Scheduled fixtures use pre-match prediction; live fixtures use pre-match plus live-state updates.
- **All live**: shows all live fixtures across competitions.

The sidebar also lets you adjust the World Cup league ID and season if your API-Football plan/account uses a different competition mapping.

If API-Football rejects the selected season on the current plan, the app automatically tries accessible fallback seasons `2022`, `2023`, and `2024`, then shows which season is actually displayed.

Dashboard tabs:

- **Match Board**: World Cup fixtures or live fixtures with prediction columns.
- **Calendar**: match-day schedule grouped by Malaysia Time (MYT, UTC+8).
- **Pre-Match Prediction**: home/draw/away probabilities from local Elo-style ratings.
- **Live Prediction**: next-goal and remaining-goal outputs when live data exists.
- **Model Drivers**: readable reasons behind the prediction.
- **Backtest**: basic completed-fixture prediction check.
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
    api_client.py
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

The prediction stack has two layers:

- **Pre-match**: local Elo-style team ratings with fallback rating `1500`.
- **Live**: transparent Poisson live-state update using API-Football statistics/events.

The combined predictor blends the pre-match prior with live match state when a fixture is live.

Feature extraction includes:

- match state: minute, elapsed fraction, remaining fraction, score difference
- discipline: red cards, yellow cards, card differences
- attacking pressure: shots, shots on target, corners, possession, pressure share
- passing and defensive context: pass accuracy, goalkeeper saves, fouls, offsides
- event context: recent events, recent goals, penalty goals
- quality flags: data completeness score
- schedule fields: API fixture date converted to Malaysia Time for calendar display

The Poisson model adjusts expected remaining goals using:

- team pressure share and shot quality
- home advantage
- score-state incentives
- red/yellow card effects
- match tempo
- remaining match time

`src/model.py` still includes a placeholder adapter for a future scikit-learn model.

`src/predictor.py` is the main dashboard-facing prediction API. It returns prediction mode, probabilities, confidence, expected goals, and readable model drivers.

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
- The active provider is API-Football only. Opta, Sportradar, SportMonks, databases, deployments, and paid-provider integrations are intentionally out of scope for the current individual-budget version.
