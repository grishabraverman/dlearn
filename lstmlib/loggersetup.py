import json
import logging
import logging.config
from pathlib import Path

CONFIG_FILE = Path(__file__).with_name("loger_congig.json")

log = None

def setup_logging(config_path: Path | str = CONFIG_FILE) -> None:
    """Load dictConfig from a JSON file and prime the logging system."""
    with open(config_path, "r", encoding="utf-8") as fh:
        config = json.load(fh)
    logging.config.dictConfig(config)