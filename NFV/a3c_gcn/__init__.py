"""EO node-only A3C-GCN training package.

Neural modules are intentionally not imported here, so the package and its
database-independent environment remain importable before PyTorch is loaded.
"""

from .training_env import DatabaseSnapshot, EOTrainingEnvironment

__all__ = ["DatabaseSnapshot", "EOTrainingEnvironment"]
