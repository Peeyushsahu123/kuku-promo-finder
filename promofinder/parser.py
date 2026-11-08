"""Parse timestamped microdrama screenplays into structured events, with explicit diagnostics.

Expected format (tolerant to case / spacing):
  EPISODE NUMBER / EPISODE LOGLINE / CONTINUITY NOTES header
  SCENE nn / Scene heading / Scene start / Scene end
  **SPEAKER (tone):**   Start: (mm:ss.ss) | End: (mm:ss.ss)   "line"
  **CAMERA:** / **ACTION:** / **SOUND FX:**   Start/End + description

Design rules (no silent repair):
  * a malformed or reversed timestamp is NEVER turned into a zero-length event or guessed; the event is
    recorded in `Episode.invalid_events` and a Diagnostic explains why;
  * overlapping tracks (camera/action/dialogue at the same time) are legitimate and kept as they are;
    overlapping *dialogue* is reported as info, not changed;
  * every file that cannot be used is reported, never dropped quietly.
"""
from __future__ import annotations
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

TS = r"(\d+):(\d+(?:\.\d+)?)"
RE_BLOCK = re.compile(r"^\*\*(.+?)\s*:?\*\*\s*:?\s*$")
RE_TIMES = re.compile(rf"^Start:\s*\(?{TS}\)?\s*\|\s*End:\s*\(?{TS}\)?\s*$", re.I)
RE_EPNUM = re.compile(r"EPISODE NUMBER:\s*(?:Episode\s*)?(\d+)", re.I)
RE_SCENE_START = re.compile(rf"Scene start:\s*{TS}", re.I)
RE_SCENE_END = re.compile(rf"Scene end:\s*{TS}", re.I)
RE_HEADING = re.compile(r"Scene heading:\s*(.+)", re.I)
NON_SPEECH = {"CAMERA", "ACTION", "SOUND FX", "SFX", "SOUND", "MUSIC", "TEXT", "TEXT ON SCREEN", "SUPER", "TITLE"}
EVENT_TOLERANCE_PAST_SCENE = 0.5     # seconds an event may run past its scene end before we warn


def fmt(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


@dataclass
class Diagnostic:
    severity: str          # error | warning | info
    code: str
    message: str
    file: str = ""
    line: int | None = None
    episode: int | None = None

    def as_dict(self):
        return {"severity": self.severity, "code": self.code, "message": self.message, "file": self.file,
                "line": self.line, "episode": self.episode}


@dataclass
class Event:
    kind: str            # dialogue | action | camera | sfx
    start: float
    end: float
    text: str
    speaker: str = ""
    tone: str = ""
    scene: int = 0
    line: int = 0


@dataclass
class Scene:
    idx: int
    heading: str
    start: float | None
    end: float | None


@dataclass
class Episode:
    num: int
    path: str
    logline: str = ""
    continuity: str = ""
    duration: float = 0.0
    scenes: list = field(default_factory=list)
    events: list = field(default_factory=list)            # events with valid timing only
    invalid_events: list = field(default_factory=list)    # (line, label, reason): excluded, never repaired
    diagnostics: list = field(default_factory=list)

    def dialogue_speakers(self):
        return sorted({e.speaker for e in self.events if e.kind == "dialogue"})

    @property
    def n_errors(self):
        return sum(1 for d in self.diagnostics if d.severity == "error")


@dataclass
class LoadResult:
    episodes: list
    diagnostics: list
    files_seen: int = 0

    @property
    def errors(self):
        return [d for d in self.diagnostics if d.severity == "error"]


def _canon(name: str) -> str:
    n = re.sub(r"\s+", " ", name).strip().upper()
    return re.sub(r"^(LADY|LORD|THE|SIR|DUKE|DUCHESS)\s+", "", n)


def _ts(m, i=1):
    """(seconds, error_message|None). A seconds field >= 60 is ambiguous, so it is an error, not a guess."""
    mm, ss = int(m.group(i)), float(m.group(i + 1))
    if not math.isfinite(ss) or ss >= 60.0:
        return None, f"seconds field {m.group(i + 1)} is not < 60"
    return mm * 60 + ss, None


def parse_episode(path: str | Path) -> Episode:
    path = Path(path)
    name = path.name
    ep = Episode(num=-1, path=str(path))

    def diag(sev, code, msg, line=None):
        ep.diagnostics.append(Diagnostic(sev, code, msg, name, line, ep.num if ep.num >= 0 else None))

    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        diag("error", "bad_encoding", f"not valid UTF-8 ({exc.reason} at byte {exc.start})")
        return ep
    except OSError as exc:
        diag("error", "unreadable", str(exc))
        return ep

    head = "\n".join(lines[:8])
    m = RE_EPNUM.search(head)
    fn = re.search(r"(\d+)", path.stem)
    if m:
        ep.num = int(m.group(1))
        if fn and int(fn.group(1)) != ep.num:
            diag("warning", "episode_number_mismatch",
                 f"header says episode {ep.num}, file name suggests {int(fn.group(1))}; using the header")
    elif fn:
        ep.num = int(fn.group(1))
        diag("warning", "episode_number_from_filename", f"no 'EPISODE NUMBER:' header; using {ep.num} from the file name")
    else:
        diag("error", "no_episode_number", "no 'EPISODE NUMBER:' header and no digits in the file name")
    for ln in lines[:8]:
        if ln.upper().startswith("EPISODE LOGLINE:"):
            ep.logline = ln.split(":", 1)[1].strip()
        elif ln.upper().startswith("CONTINUITY NOTES:"):
            ep.continuity = ln.split(":", 1)[1].strip()

    cur = None
    scene_idx = 0
    n = len(lines)
    i = 0
    while i < n:
        raw = lines[i]
        ln = raw.strip()
        lineno = i + 1
        if re.match(r"^SCENE\s+\d+\s*$", ln, re.I):
            scene_idx += 1
            cur = Scene(scene_idx, "", None, None)
            ep.scenes.append(cur)
        elif cur is not None and (mh := RE_HEADING.match(ln)):
            cur.heading = mh.group(1).strip()
        elif cur is not None and (ms := RE_SCENE_START.match(ln)):
            v, err = _ts(ms)
            if err:
                diag("error", "invalid_timestamp", f"scene start: {err}", lineno)
            else:
                cur.start = v
        elif cur is not None and (me := RE_SCENE_END.match(ln)):
            v, err = _ts(me)
            if err:
                diag("error", "invalid_timestamp", f"scene end: {err}", lineno)
            else:
                cur.end = v
        elif (mb := RE_BLOCK.match(ln)):
            label = mb.group(1).strip()
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            tline = lines[j].strip() if j < n else ""
            if not tline.lower().startswith("start:"):
                ep.invalid_events.append((lineno, label, "missing Start/End line"))
                diag("error", "missing_timing", f"block '{label}' has no Start/End line; event excluded", lineno)
                i += 1
                continue
            tm = RE_TIMES.match(tline)
            if not tm:
                ep.invalid_events.append((lineno, label, "malformed Start/End line"))
                diag("error", "malformed_timing", f"block '{label}': cannot parse timing line {tline!r}; event excluded", j + 1)
                i = j + 1
                continue
            s_val, s_err = _ts(tm, 1)
            e_val, e_err = _ts(tm, 3)
            k = j + 1
            body = []
            while k < n and lines[k].strip() and not RE_BLOCK.match(lines[k].strip()) \
                    and not lines[k].strip().startswith("--- SCRIPT END"):
                body.append(lines[k].strip())
                k += 1
            if s_err or e_err:
                ep.invalid_events.append((lineno, label, s_err or e_err))
                diag("error", "invalid_timestamp", f"block '{label}': {s_err or e_err}; event excluded", j + 1)
                i = k
                continue
            if e_val < s_val:
                ep.invalid_events.append((lineno, label, f"reversed timing {tm.group(0)}"))
                diag("error", "reversed_timing",
                     f"block '{label}': end ({e_val}) is before start ({s_val}); event excluded, not repaired", j + 1)
                i = k
                continue
            if e_val == s_val:
                diag("warning", "zero_length_event", f"block '{label}' has zero duration at {s_val}", j + 1)
            text = " ".join(body).strip()
            base = label.split("(", 1)[0].strip()
            parens = re.findall(r"\(([^()]*)\)", label)
            vo = any(re.match(r"^(V\.?O\.?|O\.?S\.?|OFF)", x.strip(), re.I) for x in parens)
            tone = ", ".join(x.strip() for x in parens if not re.match(r"^(V\.?O\.?|O\.?S\.?)$", x.strip(), re.I))
            up = base.upper()
            if up == "CAMERA":
                kind, speaker = "camera", ""
            elif up == "ACTION":
                kind, speaker = "action", ""
            elif up in NON_SPEECH or up.startswith("SOUND"):
                kind, speaker = "sfx", ""
            else:
                kind, speaker = "dialogue", _canon(base)
                text = text.strip('"“”')
                if vo:
                    tone = "[V.O.] " + tone
            if kind == "dialogue" and not text:
                diag("warning", "empty_dialogue", f"dialogue block '{label}' has no text", lineno)
            ep.events.append(Event(kind, s_val, e_val, text, speaker, tone, scene_idx, lineno))
            i = k
            continue
        elif ln.lower().startswith("start:") and RE_TIMES.match(ln) is None:
            diag("error", "malformed_timing", f"orphan timing line {ln!r}", lineno)
        i += 1

    if not ep.scenes:
        diag("error", "no_scenes", "no 'SCENE nn' markers found")
    for sc in ep.scenes:
        if sc.start is None or sc.end is None:
            diag("warning", "scene_missing_times", f"scene {sc.idx} lacks a start or end time")
        elif sc.end < sc.start:
            diag("error", "scene_reversed", f"scene {sc.idx}: end {sc.end} before start {sc.start}")
    if not ep.events:
        diag("error", "no_events", "no events with valid timing could be parsed")
    ep.events.sort(key=lambda e: (e.start, e.end))
    scene_end = {s.idx: s.end for s in ep.scenes if s.end is not None}
    beyond = [e for e in ep.events if e.scene in scene_end and e.end > scene_end[e.scene] + EVENT_TOLERANCE_PAST_SCENE]
    if beyond:
        diag("warning", "event_beyond_scene_end",
             f"{len(beyond)} event(s) end more than {EVENT_TOLERANCE_PAST_SCENE}s after their scene end "
             f"(first at line {beyond[0].line}); timings kept as written")
    dia = sorted([e for e in ep.events if e.kind == "dialogue"], key=lambda e: (e.start, e.end))
    ov = sum(1 for a, b in zip(dia, dia[1:]) if b.start < a.end - 1e-6)
    if ov:
        diag("info", "overlapping_dialogue", f"{ov} pair(s) of consecutive dialogue lines overlap in time (kept as written)")
    ep.duration = max([s.end for s in ep.scenes if s.end is not None] + [e.end for e in ep.events] + [0.0])
    return ep


def load_pack(path: str | Path) -> LoadResult:
    p = Path(path)
    diags: list[Diagnostic] = []
    if not p.exists():
        diags.append(Diagnostic("error", "input_not_found", f"{p} does not exist"))
        return LoadResult([], diags, 0)
    files = [p] if p.is_file() else sorted(p.rglob("*.txt"))
    if not files:
        diags.append(Diagnostic("error", "no_input_files", f"no .txt files under {p}"))
        return LoadResult([], diags, 0)
    parsed = [parse_episode(f) for f in files]
    for e in parsed:
        diags.extend(e.diagnostics)
    # duplicates: ambiguous, so none of the files claiming that number is used
    by_num: dict[int, list] = {}
    for e in parsed:
        if e.num >= 0:
            by_num.setdefault(e.num, []).append(e)
    dup = {n: es for n, es in by_num.items() if len(es) > 1}
    for n, es in dup.items():
        diags.append(Diagnostic("error", "duplicate_episode",
                                f"episode {n} appears in {len(es)} files: {', '.join(Path(x.path).name for x in es)}; none used",
                                episode=n))
    usable = [e for e in parsed if e.num >= 0 and e.num not in dup and e.events and e.n_errors == 0]
    skipped = [e for e in parsed if e not in usable]
    for e in skipped:
        reasons = sorted({d.code for d in e.diagnostics if d.severity == "error"}) or ["duplicate or unnumbered"]
        diags.append(Diagnostic("warning", "episode_skipped", f"{Path(e.path).name} not used: {', '.join(reasons)}",
                                Path(e.path).name, None, e.num if e.num >= 0 else None))
    nums = sorted({e.num for e in parsed if e.num >= 0})
    if nums:
        missing = sorted(set(range(nums[0], nums[-1] + 1)) - set(nums))
        if missing:
            diags.append(Diagnostic("warning", "missing_episode_numbers",
                                    f"episode numbers missing between {nums[0]} and {nums[-1]}: {missing}"))
    usable.sort(key=lambda e: e.num)
    return LoadResult(usable, diags, len(files))
