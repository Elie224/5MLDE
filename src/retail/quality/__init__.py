"""Contrôles qualité bloquants : un lot invalide arrête le pipeline, sans entraînement."""

from retail.quality.errors import QualityError
from retail.quality.validate import run_invalid_demo, validate_all, validate_batch

__all__ = ["QualityError", "run_invalid_demo", "validate_all", "validate_batch"]
