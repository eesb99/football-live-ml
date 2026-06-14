from src.schedule import fixture_myt_fields, to_myt


def test_to_myt_converts_utc_iso_datetime():
    converted = to_myt("2026-06-14T20:00:00+00:00")

    assert converted.strftime("%Y-%m-%d %H:%M") == "2026-06-15 04:00"
    assert converted.tzname() == "+08"


def test_fixture_myt_fields_handles_missing_date():
    fields = fixture_myt_fields({"fixture": {"date": None}})

    assert fields["myt_datetime"] == "TBD"
    assert fields["timezone"] == "MYT"


def test_fixture_myt_fields_formats_schedule_columns():
    fields = fixture_myt_fields(
        {"fixture": {"date": "2026-06-14T20:00:00+00:00"}}
    )

    assert fields["myt_datetime"] == "2026-06-15 04:00"
    assert fields["myt_date"] == "2026-06-15"
    assert fields["myt_time"] == "04:00"
    assert fields["timezone"] == "MYT"
