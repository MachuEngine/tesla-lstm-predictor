import yaml
import logging
import os

def load_config(config_path):
    with open(config_path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(log_file, level=logging.INFO):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )