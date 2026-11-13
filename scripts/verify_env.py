"""Thin CLI wrapper: python scripts/verify_env.py --help"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codegenai.env_check import main  # noqa: E402

if __name__ == "__main__":
    main()
