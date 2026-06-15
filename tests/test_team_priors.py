from src.ratings import TeamRating
from src.team_priors import (
    TeamPrior,
    eligible_team_prior,
    load_team_priors,
    prior_adjusted_rating,
    team_prior_is_non_leaky,
)


def test_load_team_priors_from_schema_csv(tmp_path):
    path = tmp_path / "team_priors.csv"
    path.write_text(
        "\n".join(
            [
                "team_id,team_name,strength_rating,source,source_category,as_of,available_before_kickoff,confederation,confederation_strength,host_adjustment_elo,rank",
                "10,Alpha,1700,fixture-test,pre_tournament_rating,2026-06-01T00:00:00+00:00,true,UEFA,10,0,4",
            ]
        )
        + "\n"
    )

    priors = load_team_priors(path)

    assert priors[10].team_name == "Alpha"
    assert priors[10].effective_rating == 1710
    assert priors[10].rank == 4


def test_team_prior_non_leak_guard_blocks_late_or_result_based_sources():
    clean = TeamPrior(
        team_id=10,
        team_name="Alpha",
        strength_rating=1700,
        source="fixture-test",
        source_category="pre_tournament_rating",
        as_of="2026-06-01T00:00:00+00:00",
    )
    late = TeamPrior(
        team_id=10,
        team_name="Alpha",
        strength_rating=1700,
        source="fixture-test",
        source_category="pre_tournament_rating",
        as_of="2026-06-15T00:00:00+00:00",
    )
    leaky = TeamPrior(
        team_id=10,
        team_name="Alpha",
        strength_rating=1700,
        source="fixture-test",
        source_category="same_tournament_results",
        as_of="2026-06-01T00:00:00+00:00",
    )
    kickoff = "2026-06-14T20:00:00+00:00"

    assert team_prior_is_non_leaky(clean, kickoff)
    assert not team_prior_is_non_leaky(late, kickoff)
    assert not team_prior_is_non_leaky(leaky, kickoff)
    assert eligible_team_prior({10: late}, 10, kickoff) is None


def test_prior_adjusted_rating_is_capped_and_reduces_as_results_accumulate():
    prior = TeamPrior(
        team_id=10,
        team_name="Alpha",
        strength_rating=1900,
        source="fixture-test",
        source_category="pre_tournament_rating",
        as_of="2026-06-01T00:00:00+00:00",
    )

    cold_rating, cold_delta = prior_adjusted_rating(
        TeamRating(team_id=10, team_name="Alpha", rating=1500, matches_played=0),
        prior,
    )
    deep_rating, deep_delta = prior_adjusted_rating(
        TeamRating(team_id=10, team_name="Alpha", rating=1500, matches_played=6),
        prior,
    )

    assert cold_delta == 120
    assert cold_rating == 1620
    assert deep_delta < cold_delta
    assert deep_rating < cold_rating
