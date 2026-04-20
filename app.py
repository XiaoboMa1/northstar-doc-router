"""Entry point: load config + .env, run pipeline, save, print metrics summary."""

from __future__ import annotations

import json
import logging
import os
import sys

from dotenv import load_dotenv

from file_store import write_files
from metrics import MetricsCollector
from pipeline import run


logger = logging.getLogger(__name__)


REQUIRED_CONFIG_FIELDS = (
    "docs_dir",
    "miscellaneous_file",
    "urgent_file",
    "human_review_file",
    "runtime_metadata_file",
    "keywords",
    "keyword_scoring",
    "extraction_schemas",
    "reconciliation",
    "batching",
    "llm",
)


def load_config(path: str = "config.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    missing = [k for k in REQUIRED_CONFIG_FIELDS if k not in config]
    if missing:
        raise SystemExit(f"ConfigError: Missing required config field: {missing}")
    return config


def _require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise SystemExit(f"{var} not set")
    return val


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()
    _require_env("OPENAI_API_KEY")
    _require_env("MODEL_NAME")

    config = load_config()
    metrics = MetricsCollector()

    from openai import OpenAI
    base_url = os.environ.get("OPENAI_BASE_URL") or None
    client = OpenAI(base_url=base_url) if base_url else OpenAI()

    records, run_metadata = run(config, metrics, client=client)
    write_files(records, run_metadata, config)
    metrics.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
