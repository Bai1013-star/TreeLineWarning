"""Versioned risk-policy contract, separate from science evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from core.stage_a.models import MetricDistance


class RiskClass(str, Enum):
    DANGER = "DANGER"
    WARNING = "WARNING"
    SAFE = "SAFE"


class ReviewState(str, Enum):
    VALID = "VALID"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    risk: RiskClass
    review: ReviewState
    clearance_m: float
    contact_evidence: bool | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk": self.risk.value,
            "review": self.review.value,
            "clearance_m": self.clearance_m,
            "contact_evidence": self.contact_evidence,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RiskPolicy:
    risk_policy_version: str = "risk-policy-v1"
    warning_clearance_threshold_m: float = 0.5

    def __post_init__(self) -> None:
        if type(self.risk_policy_version) is not str or not self.risk_policy_version.strip():
            raise ValueError("risk_policy_version must be a non-empty string")
        threshold = float(self.warning_clearance_threshold_m)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("warning_clearance_threshold_m must be finite and positive")
        object.__setattr__(self, "warning_clearance_threshold_m", threshold)

    def classify(
        self, *, clearance_m: float, contact: bool | None, review_required: bool = False
    ) -> RiskAssessment:
        """Apply policy to an already-metric distance; this does not detect CONTACT."""

        clearance = float(clearance_m)
        if not math.isfinite(clearance):
            raise ValueError("clearance_m must be finite")
        if contact is True:
            risk = RiskClass.DANGER
            reason = "CONTACT evidence is true"
        elif contact is False and 0.0 < clearance <= self.warning_clearance_threshold_m:
            risk = RiskClass.WARNING
            reason = "clearance is at or below the configured warning threshold"
        elif contact is False and clearance > self.warning_clearance_threshold_m:
            risk = RiskClass.SAFE
            reason = "clearance is above the configured warning threshold"
        else:
            # Unknown contact evidence is never promoted to SAFE.  The risk
            # class remains a conservative product result while quality state
            # tells the user that scientific review is still needed.
            risk = RiskClass.DANGER if clearance <= 0.0 else RiskClass.WARNING
            reason = "CONTACT evidence is unavailable; final claim requires review"
            review_required = True
        return RiskAssessment(
            risk=risk,
            review=ReviewState.REVIEW_REQUIRED if review_required else ReviewState.VALID,
            clearance_m=clearance,
            contact_evidence=contact,
            reason=reason,
        )

    def classify_metric(
        self,
        distance: MetricDistance,
        *,
        contact: bool | None,
        review_required: bool = False,
    ) -> RiskAssessment:
        """Apply policy only after a unit-evidenced conversion to metres."""

        if type(distance) is not MetricDistance:
            raise ValueError("distance must be a MetricDistance")
        return self.classify(
            clearance_m=distance.value_m,
            contact=contact,
            review_required=review_required,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_policy_version": self.risk_policy_version,
            "warning_clearance_threshold_m": self.warning_clearance_threshold_m,
        }


__all__ = ["RiskAssessment", "RiskClass", "RiskPolicy", "ReviewState"]
