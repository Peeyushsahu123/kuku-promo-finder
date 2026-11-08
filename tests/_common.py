import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from promofinder.parser import parse_episode  # noqa: E402
from promofinder.synthetic import episode_text  # noqa: E402
from promofinder.validate import build_indexes  # noqa: E402


def write(folder, name, text):
    p = Path(folder) / name
    p.write_text(text, encoding="utf-8")
    return p


def episode_from(events, num=1, scene_end=None):
    """Parse a synthetic episode built from `events` (see synthetic.episode_text)."""
    d = tempfile.mkdtemp()
    return parse_episode(write(d, f"ep{num:03d}.txt", episode_text(num, events, scene_end)))


def idx_for(ep):
    return build_indexes([ep])


def filler(a, b, step=5.0, skip=()):
    """Dialogue lines every `step` s in [a, b), 2.5 s long, avoiding any (lo, hi) range in `skip`."""
    out, t, i = [], a, 0
    while t + 2.5 <= b:
        if not any(t < hi and t + 2.5 > lo for lo, hi in skip):
            out.append(("SPEAKER" + ("A" if i % 2 == 0 else "B"), t, t + 2.5, f"Invented filler line {i}."))
        t += step
        i += 1
    return out


CLEAN_ENV = {k: v for k, v in os.environ.items()
             if k not in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "PROMO_LLM_API_KEY",
                          "PROMO_LLM_PROVIDER", "PROMO_LLM_MODEL", "PROMO_LLM_BASE_URL")}
