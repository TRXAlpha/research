"""Endogenous affective metacontrol research scaffold."""

from .affect import ConstitutiveAffectSystem
from .models import AffectState, Appraisal, CognitiveModulation, HomeostaticState
from .modulation import derive_modulation

__all__ = [
    "AffectState",
    "Appraisal",
    "CognitiveModulation",
    "ConstitutiveAffectSystem",
    "HomeostaticState",
    "derive_modulation",
]

__version__ = "0.1.0"

