"""Athlete profile (age, weight, VO2max, known HR values) from `athlete.toml` and/or the command line."""

import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

DEFAULT_FILE = Path("athlete.toml")
AGE_DECLINE_START = 35.0
AGE_DECLINE_PER_YEAR = 0.007  # endurance performance loss per year after 35 (Tanaka & Seals, 2008)
FITNESS_CAP = 0.15
_RANGES = {
    "age": (10, 100), "weight": (30, 200), "ref_weight": (30, 200),
    "vo2max": (20, 95), "ref_vo2max": (20, 95),
    "hrmax": (100, 230), "lthr": (80, 220), "hrrest": (30, 110),
}


@dataclass
class Profile:
    age: float | None = None
    weight: float | None = None  # kg, today
    vo2max: float | None = None  # ml/kg/min, today
    ref_weight: float | None = None  # kg, when the reference activities were recorded
    ref_vo2max: float | None = None
    hrmax: float | None = None
    lthr: float | None = None
    hrrest: float | None = None

    def fitness_multiplier(self) -> float:
        """Race pace multiplier for the change since the reference period (pace ~ 1 / VO2max per kg).

        A VO2max pair already includes the weight change, so the weight pair is only used without it.
        """
        if self.vo2max and self.ref_vo2max:
            ratio = self.ref_vo2max / self.vo2max
        elif self.weight and self.ref_weight:
            ratio = self.weight / self.ref_weight
        else:
            return 1.0
        return min(max(ratio, 1.0 - FITNESS_CAP), 1.0 + FITNESS_CAP)

    def age_multiplier(self, years_before_race: float) -> float:
        """Normalisation of an activity recorded `years_before_race` ago to the current age."""
        if not self.age or years_before_race <= 0:
            return 1.0
        aged = self.age - max(AGE_DECLINE_START, self.age - years_before_race)
        return 1.0 / (1.0 + AGE_DECLINE_PER_YEAR * max(aged, 0.0))


def tanaka_hrmax(age: float) -> float:
    return 208.0 - 0.7 * age


def load_profile(path=None, **overrides) -> Profile:
    """TOML values (keys = Profile fields), then non-None `overrides` (command line)."""
    data = {}
    file = Path(path) if path else DEFAULT_FILE
    if file.is_file():
        with file.open("rb") as f:
            data = tomllib.load(f)
    elif path:
        raise FileNotFoundError(f"athlete profile not found: {file}")
    unknown = set(data) - {f.name for f in fields(Profile)}
    if unknown:
        raise ValueError(f"{file}: unknown keys {sorted(unknown)}")
    values = {**data, **{k: v for k, v in overrides.items() if v is not None}}
    for key, value in values.items():
        lo, hi = _RANGES[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"athlete {key}={value!r} outside [{lo}, {hi}]")
    return Profile(**{k: float(v) for k, v in values.items()})
