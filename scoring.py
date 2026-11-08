"""Rule-based candidate generation + scoring of promo windows.

Everything here is deterministic, free and runs offline. Each promo candidate is scored on the
three things the brief asks for (hook, cliffhanger, stands-alone) plus escalation and pacing.
Scores are in [0, 1]; the weights live in WEIGHTS so they are easy to tune.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from itertools import product

from .parser import Episode, Event, fmt
from .validate import EpisodeIndex, validate_segments, MIN_LEN as _V_MIN, MAX_LEN as _V_MAX

MIN_LEN, MAX_LEN, IDEAL_LO, IDEAL_HI = _V_MIN, _V_MAX, 28.0, 52.0
HOOK_SECS = 3.0

WEIGHTS = dict(hook=0.28, cliff=0.28, alone=0.28, esc=0.10, pace=0.06)

# --- lexicons (weights are rough "how scroll-stopping is this word" values) --------------------
W_HIGH = r"""slap\w*|struck|strik\w+|punch\w*|dead|death|die[sd]?|kill\w*|murder\w*|poison\w*|blood\w*|bleed\w*|
scream\w*|explo\w+|detonat\w+|sword|dagger|blade|chain\w*|execut\w+|gallows|hang\w*|divorc\w+|pregnan\w+|
traitor\w*|betray\w*|liar|lies|lied|secret\w*|truth|reveal\w*|imposter|impostor|fraud|forged|forgery|
how\ dare|get\ out|arrest\w*|guards?|witch\w*|curse\w*|burn\w*|ransom|hostage|abduct\w*|kidnap\w*|gaol|dungeon|
forbidden|confess\w*|you\ will\ pay|you'?ll\ pay|never\ forgive|monster|thief|stole\w*|shatter\w*|crash\w*|
collaps\w+|faint\w*|seiz\w+|drag\w*|hurl\w*|shov\w+|throw\w*|threw|rip\w*|tear[sz]?|tore|whip\w*|
duchess|duke|heir|crown|royal|kneel\w*|beg\w*|spit\w*|laugh\w+\ at|humiliat\w+|disgrace\w*|banish\w*|
disown\w*|cast\ out|ruin\w*|destroy\w*|doomed|sentence\w*|trial|verdict|innocent|guilty|proof|evidence|
bastard|whore|wretch\w*|worthless|pathetic"""
RE_HIGH = re.compile(r"\b(?:" + re.sub(r"\s*\n\s*", "", W_HIGH) + r")\b", re.I)
RE_TONE_HI = re.compile(
    r"furious|enrag\w+|cold|menac\w+|shock\w*|gasp\w*|stunn\w+|horrif\w+|trembl\w+|devastat\w+|defiant|"
    r"threat\w+|chill\w+|ominous|lethal\w*|venom\w+|cruel|triumph\w+|smug|mock\w+|desperat\w+|panic\w*|"
    r"sneer\w*|vicious|savage|seeth\w+|icy|dark|sinister|hysteric\w*|scream\w*|shriek\w*|bellow\w*|"
    r"commanding|unyielding|condemn\w+|accus\w+|outrag\w+|incredulous|disbelie\w+|broken|sob\w*|weep\w*", re.I)
RE_ACTION_HI = re.compile(
    r"slap\w*|strik\w+|struck|grab\w*|seiz\w+|drag\w*|shov\w+|hurl\w*|throw\w*|threw|rip\w*|tear\w*|"
    r"burst\w*|slam\w*|smash\w*|shatter\w*|collaps\w+|crash\w*|draw[sn]?\s+(?:a\s+)?(?:sword|dagger|blade)|"
    r"blood\w*|explo\w+|fireball|flames|chains?|gasp\w*|freez\w+|stagger\w*|recoil\w*|bows?\s+her\s+head|"
    r"kneel\w*|drops?\s+to|steps?\s+forward|reveal\w*|unveil\w*|removes?\s+(?:his|her|the)\s+(?:mask|hood|veil)|"
    r"emerg\w+|silhouett\w+|appears?", re.I)
RE_REVEAL = re.compile(
    r"\b(?:you\s+(?:were|killed|knew|lied|did\s+this|are\s+(?:not|nothing|mine|dead|the\s+\w+\s+of))|it\s+was\s+(?:you|me|him|her)|i\s+am\s+(?:the|your|not)|"
    r"not\s+(?:your|my|his|her)\s+\w+|never\s+(?:was|were|be)|the\s+truth|all\s+along|"
    r"you'?re\s+(?:not|nothing|mine|dead|finished)|she'?s?\s+(?:not|alive|dead|pregnant)|"
    r"he'?s?\s+(?:not|alive|dead|the)|who\s+are\s+you|what\s+have\s+you|what\s+did\s+you|"
    r"impossible|i\s+will\s+(?:destroy|kill|end|make)|you\s+will\s+(?:pay|regret|die|hang|burn)|"
    r"lawful\s+wife|not\s+.*\s+lawful|no\s+longer)\b", re.I)
RE_CLOSURE = re.compile(
    r"\b(?:thank\s+you|thanks|goodbye|farewell|good\s+night|sleep|rest\s+now|at\s+last|finally|it'?s\s+over|"
    r"peace|forgive\s+(?:me|you)|safe\s+now|i\s+promise|together|smiles?\s+(?:warmly|softly|gently|tender\w*)|"
    r"embrac\w+|warm\s+(?:crackle|glow)|tender\w*|serene|contentment|reflective|quiet\s+resolve|"
    r"i\s+trust\s+you|every\s+day,?\s+he|i\s+know\s+almost\s+nothing)\b", re.I)
RE_SFX_CLIFF = re.compile(r"cliffhanger|unresolved|ominous|sting|stinger|sharp\s+(?:chord|crash)|"
                          r"cut\s+to\s+black|abrupt|silence\s+falls|swells?\s+to|suspens\w+|dread", re.I)
RE_FILLER_OPEN = re.compile(r"^(?:yes|no|well|so|and|but|then|because|or|thank|thanks|okay|ok|hello|good\s+(?:morning|evening)|"
                            r"i\s+see|very\s+well|perhaps|indeed)\b", re.I)
RE_DEIXIS = re.compile(r"^(?:he|she|they|it|this|that|those|these|him|her|them)\b", re.I)
RE_BACKREF = re.compile(r"\b(?:as\s+i\s+(?:said|told)|again|earlier|last\s+night|yesterday|remember\s+when|"
                        r"you\s+promised|like\s+before|that\s+(?:man|woman|night|day)|the\s+other\s+day|"
                        r"you\s+told\s+me|we\s+agreed|back\s+then)\b", re.I)
RE_PREMISE = re.compile(r"\b(?:husband|wife|marry|married|marriage|wedding|bride|groom|betroth\w+|fianc\w+|duke|duchess|"
                        r"stepmother|stepsister|sister|mother|father|brother|inherit\w+|heir|lord|lady|crown|king|queen|"
                        r"soldier|war|estate|title|dowry|vow\w*)\b", re.I)


@dataclass
class Segment:
    ep: int
    start: float
    end: float

    @property
    def dur(self):
        return self.end - self.start


@dataclass
class Candidate:
    segments: list                 # list[Segment], in playback order
    score: float = 0.0
    parts: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def dur(self):
        return sum(s.dur for s in self.segments)

    @property
    def ep(self):
        return self.segments[0].ep


# --- helpers ---------------------------------------------------------------------------------
def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _speech_events(ep: Episode):
    cached = getattr(ep, "_speech", None)
    if cached is None:
        cached = [e for e in ep.events if e.kind in ("dialogue", "action", "sfx")]
        ep._speech = cached
    return cached


def _event_intensity(e: Event) -> float:
    """0..~1.5 : how charged is this single beat (cached on the event)."""
    v = getattr(e, "_int", None)
    if v is None:
        v = e._int = _event_intensity_raw(e)
    return v


def _event_intensity_raw(e: Event) -> float:
    t = e.text
    s = 0.0
    s += 0.35 * min(3, len(RE_HIGH.findall(t)))
    s += 0.12 * t.count("!") + 0.10 * t.count("?")
    if e.kind == "dialogue":
        s += 0.25 * min(2, len(RE_TONE_HI.findall(e.tone)))
        s += 0.4 * min(2, len(RE_REVEAL.findall(t)))
        w = len(t.split())
        if w <= 7:
            s += 0.1           # punchy
    elif e.kind == "action":
        s += 0.3 * min(3, len(RE_ACTION_HI.findall(t)))
    elif e.kind == "sfx":
        s += 0.2 * min(2, len(RE_ACTION_HI.findall(t)))
    return min(s, 1.5)


def _window_events(ep: Episode, a: float, b: float):
    return [e for e in _speech_events(ep) if e.end > a + 0.05 and e.start < b - 0.05]


# --- feature scorers ---------------------------------------------------------------------------
def hook_score(ep: Episode, a: float, notes: list | None = None) -> float:
    evs = [e for e in _speech_events(ep) if e.end > a and e.start < a + HOOK_SECS]
    if not evs:
        return 0.05
    first = evs[0]
    s = 0.0
    inten = max(_event_intensity(e) for e in evs)
    s += 0.55 * _clip(inten / 0.9)
    spoken = [e for e in evs if e.kind == "dialogue"]
    if spoken:
        s += 0.15
        t = spoken[0].text
        if re.search(r"[?!]", t):
            s += 0.10
        if len(t.split()) <= 9:
            s += 0.08
        if len(t.split()) <= 2 and inten < 0.4:
            s -= 0.22                       # "What?" / "Yes." - a reaction with no context is not a hook
        if RE_FILLER_OPEN.match(t) and inten < 0.5:
            s -= 0.20
        if RE_DEIXIS.match(t):
            s -= 0.12                       # "He did it" -> who?
        if t[:1].islower() or t.startswith(("...", "…")):
            s -= 0.20                       # starts mid-sentence
    elif first.kind == "sfx":
        s += 0.10 if inten > 0.3 else 0.0
    else:
        s -= 0.05                           # silent action opening is a weaker stop-scroll
    # starting more than 1.0s into an event (mid-beat) is bad: viewer sees a cut mid-speech
    if first.start < a - 1.0:
        s -= 0.15
    return _clip(s)


def cliff_score(ep: Episode, b: float, ep_end_bonus: bool = True) -> float:
    tail = [e for e in _speech_events(ep) if e.end > b - 7.0 and e.start < b + 0.05 and e.kind != "sfx"]
    if not tail:
        return 0.05
    last = tail[-1]
    sfx_near = [e for e in ep.events if e.kind == "sfx" and b - 1.0 <= e.end <= b + 4.0]
    s = 0.0
    t = last.text
    inten = max(_event_intensity(e) for e in tail[-2:])
    s += 0.35 * _clip(inten / 0.9)
    if last.kind == "dialogue":
        if t.rstrip().endswith("?"):
            s += 0.22
        elif t.rstrip().endswith(("...", "—", "-", "…")):
            s += 0.12
        if RE_REVEAL.search(t):
            s += 0.18
        if re.search(r"[!]", t):
            s += 0.05
    else:
        if RE_ACTION_HI.search(t):
            s += 0.15
        if re.search(r"\b(?:stares?|freez\w+|stunned|horror|panic|realiz\w+|eyes\s+widen|looks?\s+up|turns?\s+to)\b", t, re.I):
            s += 0.12                        # reaction shot after a reveal
    if any(RE_SFX_CLIFF.search(e.text) for e in sfx_near):
        s += 0.15
    if RE_CLOSURE.search(t):
        s -= 0.30
    # does the story keep going right after the cut? (cut *before* the answer = real cliffhanger)
    nxt = [e for e in _speech_events(ep) if e.start >= b - 0.05 and e.kind != "sfx"]
    ep_end = ep.duration - b < 3.5
    if ep_end:
        s += 0.12 if ep_end_bonus else 0.0   # series is designed to end episodes on a cliffhanger
    elif nxt and nxt[0].start - b < 3.0:
        s += 0.10
        # ended on a question/threat and next line answers it -> good
    return _clip(s + 0.05)


def alone_score(ep: Episode, segs: list[Segment]) -> float:
    evs = [e for sg in segs for e in _window_events(ep, sg.start, sg.end)]
    dia = [e for e in evs if e.kind == "dialogue"]
    if not dia:
        return 0.15
    speakers = {e.speaker for e in dia}
    s = 0.0
    n = len(speakers)
    s += {1: 0.25, 2: 0.45, 3: 0.40}.get(n, 0.15)
    first = dia[0].text
    scene_starts = {round(sc.start, 1) for sc in ep.scenes}
    if round(segs[0].start, 1) in scene_starts or segs[0].start < 1.5:
        s += 0.12
    if RE_DEIXIS.match(first):
        s -= 0.10
    back = sum(len(RE_BACKREF.findall(e.text)) for e in dia)
    s -= 0.07 * min(3, back)
    prem = sum(len(RE_PREMISE.findall(e.text)) for e in evs)
    s += 0.07 * min(4, prem)
    # lots of unnamed/pronoun-only exchange is hard to follow; reward when speakers' names/roles appear in text
    names_said = sum(1 for e in dia for sp in speakers if len(sp) > 3 and sp.split()[0].title() in e.text)
    s += 0.05 * min(3, names_said)
    # exposition-heavy clips ("Royal Matrimonial Register", "Ashburton") are hard to follow cold:
    # count distinct capitalised words that are not sentence-initial, per 100 words spoken
    words = " ".join(e.text for e in dia)
    caps = {m.group(1) for m in re.finditer(r"(?<![.!?\"]\s)(?<!^)\b([A-Z][a-z]{3,})\b", words)}
    caps -= {sp.title() for sp in speakers}
    density = len(caps) / max(1.0, len(words.split()) / 100)
    s -= 0.10 * _clip((density - 3.0) / 6.0)
    if len(segs) > 1:
        s -= 0.15 * (len(segs) - 1)         # stitched promos risk losing continuity (judge flagged jumps)
    return _clip(s + 0.12)


def escalation_score(ep: Episode, segs: list[Segment]) -> float:
    evs = [e for sg in segs for e in _window_events(ep, sg.start, sg.end) if e.kind != "camera"]
    if len(evs) < 3:
        return 0.2
    k = max(1, len(evs) // 3)
    early = sum(_event_intensity(e) for e in evs[:k]) / k
    late = sum(_event_intensity(e) for e in evs[-k:]) / k
    mean = sum(_event_intensity(e) for e in evs) / len(evs)
    peak = max(_event_intensity(e) for e in evs)
    return _clip(0.45 * _clip(mean / 0.55) + 0.30 * _clip(peak / 1.0) + 0.25 * _clip(0.5 + (late - early)))


def pace_score(ep: Episode, segs: list[Segment]) -> float:
    total = sum(sg.dur for sg in segs)
    spoken = 0.0
    dead = 0.0
    for sg in segs:
        evs = sorted([e for e in _window_events(ep, sg.start, sg.end) if e.kind in ("dialogue", "action")], key=lambda e: e.start)
        t = sg.start
        for e in evs:
            gap = e.start - t
            if gap > 3.0:
                dead += gap - 3.0
            t = max(t, e.end)
            if e.kind == "dialogue":
                spoken += min(e.end, sg.end) - max(e.start, sg.start)
        if sg.end - t > 3.0:
            dead += sg.end - t - 3.0
    density = spoken / max(total, 1e-6)
    s = _clip(density / 0.55) * 0.6 + _clip(1 - dead / 8.0) * 0.4
    return _clip(s)


def duration_factor(d: float) -> float:
    if d < MIN_LEN or d > MAX_LEN:
        return 0.0
    if IDEAL_LO <= d <= IDEAL_HI:
        return 1.0
    return 0.93 if d < IDEAL_LO else 0.95


def score_candidate(ep: Episode, cand: Candidate) -> Candidate:
    segs = cand.segments
    h = hook_score(ep, segs[0].start)
    c = cliff_score(ep, segs[-1].end)
    al = alone_score(ep, segs)
    es = escalation_score(ep, segs)
    pc = pace_score(ep, segs)
    base = (WEIGHTS["hook"] * h + WEIGHTS["cliff"] * c + WEIGHTS["alone"] * al + WEIGHTS["esc"] * es + WEIGHTS["pace"] * pc)
    # a promo needs BOTH a hook and a cliffhanger: the weaker of the two scales the whole score
    gate = 0.45 + 0.55 * min(h, c)
    cand.parts = dict(hook=h, cliff=c, alone=al, esc=es, pace=pc, base=base, gate=gate)
    cand.score = base * gate * duration_factor(cand.dur) * (0.88 if len(segs) > 1 else 1.0)
    return cand


# --- candidate generation ----------------------------------------------------------------------
def get_index(ep: Episode) -> EpisodeIndex:
    ix = getattr(ep, "_index", None)
    if ix is None:
        ix = ep._index = EpisodeIndex(ep)
    return ix


def _anchors(ep: Episode):
    """Legal cut points only: source event boundaries that do not fall strictly inside a dialogue line."""
    ix = get_index(ep)
    starts = set()
    for e in _speech_events(ep):
        if e.kind in ("dialogue", "action") or (e.kind == "sfx" and (e.start < 1.0 or (e.text and RE_ACTION_HI.search(e.text)))):
            starts.add(round(e.start, 2))
    legal_starts = set(ix.cut_starts())
    ends = {round(e.end, 2) for e in _speech_events(ep) if e.kind in ("dialogue", "action")}
    legal_ends = set(ix.cut_ends())
    st = sorted(x for x in starts | ({0.0} if 0.0 in legal_starts else set()) if x in legal_starts)
    en = sorted(x for x in ends if x in legal_ends)
    return st, en


def is_valid(ep: Episode, cand: "Candidate") -> bool:
    ix = get_index(ep)
    return not validate_segments({ep.num: ix}, [(s.ep, s.start, s.end) for s in cand.segments])


def generate_single(ep: Episode) -> list[Candidate]:
    starts, ends = _anchors(ep)
    out = []
    for a in starts:
        for b in ends:
            d = b - a
            if MIN_LEN <= d <= MAX_LEN:
                out.append(score_candidate(ep, Candidate([Segment(ep.num, a, b)])))
    return out


def generate_stitched(ep: Episode, singles_scored: list[Candidate], k: int = 14) -> list[Candidate]:
    """Two-segment promos inside one episode: a hooky opener + a cliffhanger closer with a skipped middle."""
    starts, ends = _anchors(ep)
    # opener candidates: short stretches [a,b] 8-24s ranked by hook+intensity
    openers, closers = [], []
    for a in starts:
        for b in ends:
            d = b - a
            if 8 <= d <= 24:
                evs = _window_events(ep, a, b)
                if not evs:
                    continue
                openers.append((hook_score(ep, a) * 0.7 + 0.3 * _clip(max(_event_intensity(e) for e in evs)), a, b))
            if 10 <= d <= 34:
                closers.append((cliff_score(ep, b) * 0.7 + 0.3 * _clip(max(_event_intensity(e) for e in _window_events(ep, a, b)) if _window_events(ep, a, b) else 0), a, b))
    openers = sorted(openers, reverse=True)[:k]
    closers = sorted(closers, reverse=True)[:k]
    out = []
    for (_, a1, b1), (_, a2, b2) in product(openers, closers):
        gap = a2 - b1
        if gap < 4.0 or gap > 70:           # must really skip something; not a continuous cut
            continue
        d = (b1 - a1) + (b2 - a2)
        if MIN_LEN <= d <= MAX_LEN:
            out.append(score_candidate(ep, Candidate([Segment(ep.num, a1, b1), Segment(ep.num, a2, b2)])))
    return out


def overlap(c1: Candidate, c2: Candidate) -> float:
    inter = 0.0
    for s1 in c1.segments:
        for s2 in c2.segments:
            if s1.ep == s2.ep:
                inter += max(0.0, min(s1.end, s2.end) - max(s1.start, s2.start))
    return inter / max(1e-6, min(c1.dur, c2.dur))


def pick_for_episode(cands: list[Candidate], per_ep: int, max_overlap: float = 0.25, floor: float = 0.0):
    chosen = []
    for c in sorted(cands, key=lambda c: -c.score):
        if c.score < floor:
            break
        if all(overlap(c, x) <= max_overlap for x in chosen):
            chosen.append(c)
        if len(chosen) >= per_ep:
            break
    return chosen


def find_candidates(ep: Episode, per_ep: int = 2, stitched: bool = True, floor: float = 0.0):
    singles = generate_single(ep)
    pool = list(singles)
    if stitched:
        pool += generate_stitched(ep, singles)
    pool = [c for c in pool if is_valid(ep, c)]          # shared validator: nothing illegal can be picked
    return pick_for_episode(pool, per_ep, floor=floor), len(pool)
