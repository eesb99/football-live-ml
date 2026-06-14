from src.adapters import NullPaidDataAdapter, PaidDataSnapshot, default_paid_data_snapshot


def test_null_paid_data_adapter_is_fallback_safe():
    snapshot = NullPaidDataAdapter().get_snapshot(123)

    assert snapshot.odds_available is False
    assert snapshot.real_xg_available is False
    assert snapshot.injuries_available is False
    assert snapshot.news_available is False
    assert snapshot.home_real_xg is None
    assert snapshot.availability_summary == (
        "odds=missing, real_xg=missing, injuries=missing, news=missing"
    )


def test_paid_data_snapshot_exports_prediction_fields():
    snapshot = PaidDataSnapshot(
        odds_available=True,
        odds_source="example odds",
        home_odds_implied_probability=0.5,
        draw_odds_implied_probability=0.25,
        away_odds_implied_probability=0.25,
    )

    fields = snapshot.as_prediction_fields()

    assert fields["odds_available"] is True
    assert fields["odds_source"] == "example odds"
    assert fields["home_odds_implied_probability"] == 0.5
    assert fields["paid_data_availability"] == (
        "odds=available, real_xg=missing, injuries=missing, news=missing"
    )


def test_default_paid_data_snapshot_uses_null_adapter():
    snapshot = default_paid_data_snapshot()

    assert snapshot == NullPaidDataAdapter().get_snapshot(0)
