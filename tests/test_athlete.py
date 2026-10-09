import pytest

from trailsport.athlete import Profile, load_profile
from trailsport.hr import estimate_athlete
from trailsport.metrics import minetti_cost


def test_profile_file_and_overrides(tmp_path):
    f = tmp_path / "athlete.toml"
    f.write_text("age = 40\nweight = 70.0\nvo2max = 55\n", encoding="utf-8")
    p = load_profile(f, weight=68.0, lthr=None)
    assert (p.age, p.weight, p.vo2max, p.lthr) == (40.0, 68.0, 55.0, None)


@pytest.mark.parametrize("content", ["age = 5\n", "speed = 3\n", "weight = 'heavy'\n"])
def test_profile_rejects_bad_values(tmp_path, content):
    f = tmp_path / "athlete.toml"
    f.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_profile(f)


def test_fitness_multiplier():
    assert Profile().fitness_multiplier() == 1.0
    assert Profile(vo2max=55, ref_vo2max=50).fitness_multiplier() == pytest.approx(50 / 55)
    assert Profile(weight=72, ref_weight=70).fitness_multiplier() == pytest.approx(72 / 70)
    # the VO2max pair already includes the weight change
    assert Profile(vo2max=55, ref_vo2max=55, weight=72, ref_weight=70).fitness_multiplier() == 1.0
    assert Profile(vo2max=80, ref_vo2max=40).fitness_multiplier() == pytest.approx(0.85)


def test_age_multiplier_only_after_35():
    assert Profile(age=30).age_multiplier(2) == 1.0
    assert Profile(age=50).age_multiplier(1) == pytest.approx(1 / 1.007)
    assert Profile(age=36).age_multiplier(3) == pytest.approx(1 / 1.007)


def test_hr_values_from_profile_and_age():
    ath = estimate_athlete([], Profile(age=40))
    assert ath.hrmax == pytest.approx(180) and ath.lthr == pytest.approx(162)
    assert ath.sources == {"hrmax": "age (Tanaka)", "lthr": "0.9 x HRmax"}
    ath = estimate_athlete([], Profile(age=40, hrmax=190, lthr=170))
    assert (ath.hrmax, ath.lthr) == (190, 170)


def test_minetti_cost_flat():
    assert float(minetti_cost(0.0)) == pytest.approx(3.6)
    assert float(minetti_cost(-10.0)) < 3.6 < float(minetti_cost(10.0))
