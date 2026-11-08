"""One shared validator for every exported promo segment (rule engine, stitched cuts and LLM trims alike).

Rules (all must hold):
  V1 each segment has finite start < end, start >= 0 and end <= the episode's duration;
  V2 each start is a SOURCE start (start of a dialogue / action / sfx event, or a scene start) and each end
     is a SOURCE end (end of such an event, or a scene end). Camera events are not cut points;
  V3 no boundary lies STRICTLY INSIDE a dialogue interval (so a cut never splits a spoken line);
  V4 segments are in chronological order and do not overlap (touching is allowed);
  V5 the total duration is between 20 and 60 seconds (inclusive, 0.01 s resolution).

How action / camera / sfx intervals are treated: their start and end times are legal cut points (V2), but a
cut MAY cross an action, camera or sfx interval, because those tracks overlap dialogue in these scripts and
there is no picture to check that an action is visually complete. Only spoken dialogue is protected (V3).
"""
from __future__ import annotations
import math
from dataclasses import dataclass

MIN_LEN, MAX_LEN = 20.0, 60.0
Q = 100                      # timestamps have 0.01 s resolution -> compare as integers
SOURCE_KINDS = ("dialogue", "action", "sfx")


def _q(x: float) -> int:
    return int(round(x * Q))


@dataclass(frozen=True)
class Violation:
    code: str
    message: str

    def __str__(self):
        return f"{self.code}: {self.message}"


class EpisodeIndex:
    """Precomputed cut-point tables for one episode (built once, shared by generator and validator)."""

    def __init__(self, ep):
        self.num = ep.num
        self.duration_q = _q(ep.duration)
        self.dialogue = sorted((_q(e.start), _q(e.end)) for e in ep.events if e.kind == "dialogue" and e.end > e.start)
        starts = {_q(e.start) for e in ep.events if e.kind in SOURCE_KINDS}
        ends = {_q(e.end) for e in ep.events if e.kind in SOURCE_KINDS}
        for sc in ep.scenes:
            if sc.start is not None:
                starts.add(_q(sc.start))
            if sc.end is not None:
                ends.add(_q(sc.end))
        self.source_starts = starts
        self.source_ends = ends

    def inside_dialogue(self, tq: int):
        """Return the (start, end) dialogue interval that strictly contains tq, else None."""
        for s, e in self.dialogue:
            if s < tq < e:
                return s, e
        return None

    def cut_starts(self):
        return sorted(t / Q for t in self.source_starts if t <= self.duration_q and not self.inside_dialogue(t))

    def cut_ends(self):
        return sorted(t / Q for t in self.source_ends if t <= self.duration_q and not self.inside_dialogue(t))


def validate_segments(indexes: dict, segments: list, min_len: float = MIN_LEN, max_len: float = MAX_LEN) -> list:
    """indexes: {episode_number: EpisodeIndex}; segments: [(episode, start, end), ...] in playback order."""
    out: list[Violation] = []
    if not segments:
        return [Violation("empty", "promo has no segments")]
    total_q = 0
    prev = None
    for n, (epn, a, b) in enumerate(segments):
        tag = f"segment {n + 1} (ep {epn})"
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isfinite(a) and math.isfinite(b)):
            out.append(Violation("not_finite", f"{tag}: start/end must be finite numbers"))
            continue
        ix = indexes.get(epn)
        if ix is None:
            out.append(Violation("unknown_episode", f"{tag}: episode {epn} not loaded"))
            continue
        aq, bq = _q(a), _q(b)
        if not aq < bq:
            out.append(Violation("non_positive_length", f"{tag}: start {a} is not before end {b}"))
            continue
        if aq < 0 or bq > ix.duration_q:
            out.append(Violation("out_of_episode_bounds", f"{tag}: {a}-{b} outside 0-{ix.duration_q / Q}"))
        if aq not in ix.source_starts:
            out.append(Violation("start_not_source_boundary", f"{tag}: start {a} is not the start of any dialogue/action/sfx event or scene"))
        if bq not in ix.source_ends:
            out.append(Violation("end_not_source_boundary", f"{tag}: end {b} is not the end of any dialogue/action/sfx event or scene"))
        for label, tq in (("start", aq), ("end", bq)):
            hit = ix.inside_dialogue(tq)
            if hit:
                out.append(Violation("boundary_inside_dialogue",
                                     f"{tag}: {label} {tq / Q} falls inside dialogue {hit[0] / Q}-{hit[1] / Q}"))
        if prev is not None and prev[0] == epn and aq < prev[2]:
            out.append(Violation("segments_not_chronological", f"{tag}: starts at {a} before previous segment ends at {prev[2] / Q}"))
        elif prev is not None and prev[0] > epn:
            out.append(Violation("segments_not_chronological", f"{tag}: episode {epn} comes after episode {prev[0]}"))
        prev = (epn, aq, bq)
        total_q += bq - aq
    lo, hi = _q(min_len), _q(max_len)
    if not (lo <= total_q <= hi):
        out.append(Violation("duration_out_of_range", f"total duration {total_q / Q:.2f}s not in [{min_len}, {max_len}]"))
    return out


def build_indexes(episodes) -> dict:
    return {e.num: EpisodeIndex(e) for e in episodes}
