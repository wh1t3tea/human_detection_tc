#!/usr/bin/env python3
"""
Configuration utilities for loading and parsing YAML configs.
"""

import os
from typing import Dict, Any, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file can't be parsed
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return config or {}

    except Exception as e:
        raise ValueError(f"Error loading config from {config_path}: {e}")
