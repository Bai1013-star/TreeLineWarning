"""Product-level contact evidence contract for final tree clearance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ContactState(str, Enum):
    CONTACT_CONFIRMED = "CONTACT_CONFIRMED"
    CONTACT_NOT_CONFIRMED = "CONTACT_NOT_CONFIRMED"
    CONTACT_UNCERTAIN = "CONTACT_UNCERTAIN"


@dataclass(frozen=True, slots=True)
class ContactEvidence:
    state: ContactState
    evidence: str
    review_required: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "contact_state": self.state.value,
            "contact_evidence": self.evidence,
            "review_required": self.review_required,
        }


def contact_evidence_from_clearance(
    minimum_clearance_m: float | None,
    *,
    clearance_status: str,
    tie_tolerance_m: float = 1e-9,
) -> ContactEvidence:
    """Classify evidence without treating exact zero as confirmed contact.

    The current repository has no reviewed physical-contact detector.  A
    positive, valid geometric separation supports ``NOT_CONFIRMED``; missing
    or zero/tie-level separation remains uncertain and requires review.
    """

    if clearance_status != "VALID" or minimum_clearance_m is None:
        return ContactEvidence(
            ContactState.CONTACT_UNCERTAIN,
            "validated contact detector is unavailable",
            True,
        )
    value = float(minimum_clearance_m)
    if value > tie_tolerance_m:
        return ContactEvidence(
            ContactState.CONTACT_NOT_CONFIRMED,
            "validated wire/tree geometry has positive separation",
            False,
        )
    return ContactEvidence(
        ContactState.CONTACT_UNCERTAIN,
        "zero or tie-level clearance is not sufficient contact evidence",
        True,
    )


__all__ = [
    "ContactEvidence",
    "ContactState",
    "contact_evidence_from_clearance",
]
