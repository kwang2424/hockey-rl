"""A 1v1 ice hockey environment for reinforcement learning."""

from .config import Config, DEFAULT
from .env import VecHockeyEnv, OBS_DIM, ACT_DIM, OBS_SLICES

__all__ = ["Config", "DEFAULT", "VecHockeyEnv", "OBS_DIM", "ACT_DIM", "OBS_SLICES"]
