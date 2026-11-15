"""Single entry point for the candidate-cluster depth profiles.

Formal runtime supports only Top-1 (fast) and Top-3 (balanced).  Top-8 was a
historical/high-recall research configuration and is intentionally no longer a
formal profile or CLI choice.  The low-level protocol still accepts an explicit
integer for historical experiment reproduction; formal CLIs validate the range.
"""
from __future__ import annotations

CANDIDATE_DEPTH_PROFILES = {
    "fast": 1,
    "balanced": 3,
}
DEFAULT_CANDIDATE_DEPTH_PROFILE = "balanced"
LEGACY_CANDIDATE_DEPTH_PROFILE = "fast"
FORMAL_CANDIDATE_DEPTHS = (1, 3)
FORMAL_MAX_CANDIDATE_DEPTH = 3


def resolve_candidate_depth(profile: str | int | None = None) -> int:
    """Resolve a profile name or integer depth to a positive cluster count."""
    if profile is None:
        profile = DEFAULT_CANDIDATE_DEPTH_PROFILE
    if isinstance(profile, bool):
        raise ValueError("candidate depth must be an int or profile name")
    if isinstance(profile, int):
        if profile < 1:
            raise ValueError(f"candidate depth must be >= 1, got {profile}")
        return profile
    key = str(profile).strip().lower()
    if key not in CANDIDATE_DEPTH_PROFILES:
        raise ValueError(
            f"unknown candidate depth profile {profile!r}; "
            f"choose from {sorted(CANDIDATE_DEPTH_PROFILES)} or pass an int"
        )
    return CANDIDATE_DEPTH_PROFILES[key]


def validate_formal_candidate_depth(value: int) -> int:
    """Reject depths outside the formal Top-1..Top-3 runtime range.

    Historical Top-8 experiments may still call the low-level protocol with an
    explicit integer, but no formal CLI/benchmark path may select it.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"candidate depth must be an int, got {value!r}")
    if value < 1 or value > FORMAL_MAX_CANDIDATE_DEPTH:
        raise ValueError(
            f"formal candidate depth must be in 1..{FORMAL_MAX_CANDIDATE_DEPTH}; "
            f"got {value}. Top-8 is historical/research-only."
        )
    return value
