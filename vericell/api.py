"""Meta Model API client setup. The key comes from the environment or a git-ignored .env."""

import os
import sys
from pathlib import Path

from openai import OpenAI

BASE_URL = "https://api.meta.ai/v1"
ROOT = Path(__file__).resolve().parent.parent


def load_key() -> str:
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() and not line.lstrip().startswith("#"):
                os.environ.setdefault(name.strip(), value.strip().strip("\"'"))
    key = os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY")
    if not key:
        sys.exit("No API key found. Set META_API_KEY (or MODEL_API_KEY) in your shell or in .env.")
    return key


def client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=load_key())
