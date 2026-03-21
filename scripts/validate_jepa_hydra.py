#!/usr/bin/env python
"""Validate JEPA with Hydra (same config groups as training; resolves experiment_dir)."""

from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig

_root = Path(__file__).resolve().parent.parent
_scripts = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_scripts))

from validate_jepa import run_validation_main


@hydra.main(version_base=None, config_path="../configs", config_name="validate")
def main(cfg: DictConfig) -> bool | None:
    return run_validation_main(cfg)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
