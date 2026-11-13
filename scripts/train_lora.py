"""Thin CLI wrapper: python scripts/train_lora.py --help"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codegenai.training.train import main  # noqa: E402

if __name__ == "__main__":
    main()
