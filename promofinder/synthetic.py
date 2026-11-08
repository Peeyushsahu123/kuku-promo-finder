"""Synthetic screenplays in the Kuku format, with INVENTED content, for tests and the dependency-free demo.
Nothing here comes from any real show."""
from __future__ import annotations
from pathlib import Path


def _t(x: float) -> str:
    return f"{int(x // 60):02d}:{x % 60:05.2f}"


def block(kind: str, start: float, end: float, text: str, tone: str = "") -> str:
    label = kind if kind in ("CAMERA", "ACTION", "SOUND FX") else (f"{kind} ({tone})" if tone else kind)
    body = f'"{text}"' if kind not in ("CAMERA", "ACTION", "SOUND FX") else text
    return f"**{label}:**\nStart: ({_t(start)}) | End: ({_t(end)})\n{body}\n"


def episode_text(num: int, events: list, scene_end: float | None = None, header: bool = True, logline: str = "A synthetic test episode.") -> str:
    """events: list of (kind, start, end, text[, tone]) where kind is CAMERA/ACTION/SOUND FX or a SPEAKER NAME."""
    end = scene_end if scene_end is not None else max(e[2] for e in events)
    parts = []
    if header:
        parts += [f"EPISODE NUMBER: Episode {num}", f"EPISODE LOGLINE: {logline}", "CONTINUITY NOTES: synthetic", ""]
    parts += ["--- SCRIPT START ---", "", "SCENE 01", "Scene heading: INT. LIGHTHOUSE - NIGHT", "Scene start: 00:00.00",
              f"Scene end: {_t(end)}", "", "Actions and dialogue:", ""]
    for e in events:
        parts.append(block(*e))
    parts.append("--- SCRIPT END ---")
    return "\n".join(parts) + "\n"


LINES = [("MARA", "Open the door, Tobin!", "Furious"), ("TOBIN", "You are not welcome here.", "Cold"),
         ("MARA", "You lied to the whole harbour!", "Outraged"), ("TOBIN", "Prove it.", "Menacing"),
         ("MARA", "I found the ledger in the boathouse.", "Shaken"), ("TOBIN", "Then you know what I did.", "Chilling"),
         ("MARA", "Who else has seen it?", "Whispering"), ("TOBIN", "Nobody. Yet.", "Smug")]


def demo_episode(num: int, length: float = 90.0, step: float = 5.0) -> str:
    """A well-formed synthetic episode: alternating dialogue every `step` seconds, plus camera/action tracks."""
    ev, t, i = [], 1.0, 0
    while t + 3.0 < length:
        who, text, tone = LINES[(i + num) % len(LINES)]
        ev.append((who, t, t + 3.0, text, tone))
        ev.append(("CAMERA", t, t + 3.2, "Close-up on the speaker."))
        if i % 4 == 3:
            ev.append(("ACTION", t + 3.0, t + step, f"A lantern swings and goes dark ({i})."))
        t += step
        i += 1
    ev.append(("SOUND FX", length - 2.0, length, "Ominous cliffhanger drone."))
    return episode_text(num, ev, scene_end=length)


def write_demo_pack(folder, n_episodes: int = 3) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for n in range(1, n_episodes + 1):
        (folder / f"episode-{n:03d}-synthetic.txt").write_text(demo_episode(n, length=80.0 + 10 * n), encoding="utf-8")
    return folder
