# Experiment Notes

## 2026-06-14

- User goal is an individual-budget World Cup prediction dashboard, not a betting-grade enterprise product.
- Preserve API-Football as the only active provider.
- Avoid unnecessary quota burn: cache season fixtures and fetch fixture detail only for live or selected fixtures.
- Next best action: add Elo pre-match prior, combined live predictor, model drivers, prediction snapshots, and dashboard tabs.
- Implemented Elo pre-match prior, combined predictor, model drivers, local prediction/rating snapshots, and dashboard tabs.
- Evidence so far: `python3 -m pytest tests`, compileall, and Streamlit AppTest passed during implementation.
- Next best action: run final verification sequence and confirm `http://localhost:8501`.
- Final verification evidence:
  - `python3 -m pytest tests` passed with 13 tests.
  - `python3 -m compileall src app tests` passed.
  - Streamlit AppTest printed `streamlit_app_executed`.
  - Streamlit server started on `http://localhost:8501`.
  - Escalated localhost check returned `HTTP/1.1 200 OK`.
- Added calendar and match schedule display in Malaysia Time (MYT, UTC+8).
- Latest evidence after MYT update: `python3 -m pytest tests` passed with 16 tests; compileall and Streamlit AppTest passed.
- Added API-Football free-plan season fallback so `season=2026` rejection can fall back to accessible seasons `2022`, `2023`, and `2024`.
- Latest evidence after fallback update: `python3 -m pytest tests` passed with 18 tests; compileall and Streamlit AppTest passed.
