"""Evaluation utilities: synthetic sequence generator + MOT metrics."""
from .synthetic import generate_sequence
from .metrics import evaluate_mot, MOTMetrics

__all__ = ["generate_sequence", "evaluate_mot", "MOTMetrics"]
