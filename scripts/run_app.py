"""Thin CLI wrapper: python scripts/run_app.py --help"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codegenai.app.gradio_app import main  # noqa: E402

if __name__ == "__main__":
    main()
