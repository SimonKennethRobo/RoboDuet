"""Run UMI-on-Legs through the bounded common-plant frozen-library adapter."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark.data.run_visual_library import main


if __name__ == "__main__":
    raise SystemExit(main(method="umi"))
