"""Dependency-free demo on synthetic data:  python -m promofinder.demo [out_folder]"""
import sys
import tempfile
from pathlib import Path

from .pipeline import run
from .synthetic import write_demo_pack


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demo_out")
    with tempfile.TemporaryDirectory() as d:
        pack = write_demo_pack(Path(d) / "episodes", 3)
        records, summary = run(pack, out, title="Synthetic demo")
    print(f"\n{len(records)} promos written to {out.resolve()} (open promos.html). Every segment passed the validator: "
          f"{summary['validation']}")


if __name__ == "__main__":
    main()
