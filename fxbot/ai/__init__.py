"""Isolated, non-executing ML prediction components for FX candidate evaluation.

Import concrete services from their modules so package initialization stays
lightweight and cannot create circular dependencies between schemas, feature
building, model loading, and prediction.
"""
