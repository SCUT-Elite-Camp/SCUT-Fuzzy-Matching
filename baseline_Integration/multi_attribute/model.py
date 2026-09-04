from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MatchRecord:
    """One entity record used by the first multi-attribute prototype.

    ``dob`` may be omitted. Missing attributes contribute zero score rather than
    being encoded as a literal string such as ``"null"``.
    """

    name: str
    dob: str | None = None
    record_id: str | None = None
