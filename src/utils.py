"""Small shared helpers: config loading, seeding, path resolution."""
import random
from pathlib import Path

import numpy as np
import yaml

# Project root = parent of the src/ folder, so scripts work from any cwd.
ROOT = Path(__file__).resolve().parent.parent


def load_config(path="config.yaml"):
    """Read config.yaml into a dict."""
    with open(ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """Fix all random seeds so results are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def p(rel):
    """Resolve a path from config (relative to project root) to an absolute Path."""
    return ROOT / rel
