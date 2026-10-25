"""
config.py

Central configuration for the mission planner. All constants, API keys,
and environment loading live here.
"""

import os

# Environment loading

def _load_env_file(env_path: str) -> None:
    """
    Loads key=value pairs from a .env file into os.environ.
    """
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key   = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


# Load .env from repo root on import.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
_load_env_file(os.path.join(_PROJECT_ROOT, ".env"))


# LLM configuration

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_API_KEY")
GROQ_MODEL   = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# Retry configuration

# How many times the pipeline retries a failed LLM call before giving up.
MAX_RETRIES  = 3

# Few-shot retrieval configuration

# How many few-shot examples are retrieved per instruction.
FEW_SHOT_K   = 4