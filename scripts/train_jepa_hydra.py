#!/usr/bin/env python
"""Train JEPA with Hydra configs (primary entrypoint)."""

from pathlib import Path
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

_root = Path(__file__).resolve().parent.parent
_scripts = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_scripts))

from jepa_training import run_training


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    out = Path(HydraConfig.get().runtime.output_dir)
    run_training(cfg, hydra_output_dir=out)


if __name__ == "__main__":
    main()
