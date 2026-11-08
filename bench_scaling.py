"""Measure rule-stage runtime on synthetic data: vs number of episodes N, and vs events per episode (density).
Usage: python tools/bench_scaling.py"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from promofinder.pipeline import run  # noqa: E402
from promofinder.synthetic import episode_text  # noqa: E402


def pack(n, step):
    d = Path(tempfile.mkdtemp())
    for k in range(1, n + 1):
        ev, t, i = [], 1.0, 0
        while t + step < 120:
            ev.append(("MARA" if i % 2 else "TOBIN", t, t + step * 0.8, f"Invented line number {i} with some drama!"))
            ev.append(("CAMERA", t, t + step, "Close-up."))
            t += step
            i += 1
        (d / f"ep{k:03d}.txt").write_text(episode_text(k, ev, 120.0), encoding="utf-8")
    return d


def timed(n, step):
    d = pack(n, step)
    t = time.time()
    run(d, Path(tempfile.mkdtemp()), log=lambda *_: None)
    return time.time() - t


if __name__ == "__main__":
    print("episodes N (dialogue every 5 s, ~24 lines/episode):")
    for n in (10, 20, 40, 80):
        print(f"  N={n:3d}: {timed(n, 5.0):6.2f} s")
    print("events per episode (N=10, 120 s episodes):")
    for step in (6.0, 3.0, 1.5, 0.75):
        print(f"  dialogue every {step:4.2f} s (~{int(120 / step)} lines): {timed(10, step):6.2f} s")
