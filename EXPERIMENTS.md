# World Cup Prediction Dashboard Experiments

| Attempt | Description | Result | Evidence | Decision |
|---|---|---|---|---|
| Live Poisson baseline | Existing API-Football live stats/events are converted to features and fed into a Poisson live model. | Works for in-play fixtures but cannot predict scheduled fixtures. | Existing tests pass for live probability invariants. | Keep as the live update layer. |
| Provider choice | Considered Opta/Sportradar/SportMonks versus API-Football for an individual project. | Enterprise providers are not practical for this phase. | User confirmed individual-builder context. | Keep API-Football only. |
| Elo pre-match layer | Add fallback local Elo ratings and Poisson expected goals for scheduled fixtures. | Works without historical data by falling back to 1500 ratings; improves when completed fixtures are available. | `tests/test_predictor.py` covers rating updates and pre-match probability invariants. | Keep for MVP. |
| Combined predictor | Blend pre-match prior with live Poisson output for in-play fixtures. | Produces live mode, next-goal outputs, and readable model drivers. | `tests/test_predictor.py` covers live blend behavior and red-card driver output. | Keep for MVP. |
