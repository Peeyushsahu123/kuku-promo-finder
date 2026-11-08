"""Optional LLM judge. Swappable provider, stdlib only, bounded timeouts, strict response validation.

Provider selection (env vars or CLI flags):
  PROMO_LLM_PROVIDER = gemini | openai | anthropic | none   (openai = any OpenAI-compatible endpoint)
  PROMO_LLM_MODEL, PROMO_LLM_BASE_URL, PROMO_LLM_TIMEOUT (seconds per request, default 30)
  Key: PROMO_LLM_API_KEY, or GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY.  Keys are read from the
  environment only; they are never written to outputs and are redacted from error messages.

The LLM never decides what is exported. It scores candidates the rule engine produced; anything it returns is
validated (structure, ids, finite scores in range, evidence quoted from inside the clip, trims re-checked by the
shared validator in pipeline.py). Anything invalid is rejected and the rule score is kept.
"""
from __future__ import annotations
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from .parser import Episode, fmt

DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 2                  # extra attempts after the first (429 / 5xx / network only)

SYSTEM = (
    "You are a senior social-video editor for vertical microdrama (ReelShort / DramaBox style). "
    "You review PROMO candidates: 20-60 s clips cut from the show to win NEW viewers. A great promo (1) opens on a "
    "HOOK in the first 3 seconds, (2) ends on an unresolved CLIFFHANGER, cut before the answer, and (3) makes sense "
    "to someone who has never seen the show. Be harsh. Reply with JSON only."
)

PROMPT = """SERIES: {title}
EPISODE {ep} LOGLINE: {logline}
CONTINUITY: {continuity}

FULL EPISODE TIMELINE (mm:ss, speaker or kind, text):
{timeline}

CANDIDATE PROMOS (segments are [start-end] of this episode):
{cands}

For EACH candidate id return an object with:
 "id": the candidate id (exactly as given),
 "hook": number 1-5, "cliffhanger": number 1-5, "standalone": number 1-5,
 "keep": true or false,
 "better_start": optional mm:ss.xx - an existing event start inside this episode that gives a stronger opening,
 "better_end": optional mm:ss.xx - an existing event end that gives a stronger cut,
 "hook_text": VERBATIM words copied from the script lines inside the first 3 s of the clip (<= 15 words),
 "cliffhanger_text": VERBATIM words copied from the last lines of the clip (<= 15 words),
 "why": ONE sentence (<= 40 words).
Never quote text that is not inside the clip. JSON: {{"results":[...]}}"""


class LLMError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def _redact(msg: str, key: str) -> str:
    if key:
        msg = msg.replace(key, "***")
    return re.sub(r"(key=|Bearer\s+|x-api-key:\s*)[A-Za-z0-9_\-]{8,}", r"\1***", msg)[:400]


def _is_loopback(url: str) -> bool:
    host = urllib.parse.urlparse(url).hostname or ""
    return host in ("localhost", "127.0.0.1", "::1")


def _post(url, headers, payload, timeout, retries=MAX_RETRIES, backoff=1.0, key=""):
    """POST JSON with a hard per-request timeout and a bounded number of retries.
    Loopback URLs bypass any HTTP(S)_PROXY setting (a proxy would otherwise swallow test traffic)."""
    data = json.dumps(payload).encode()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if _is_loopback(url) else urllib.request.build_opener()
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
            with opener.open(req, timeout=timeout) as r:
                body = r.read(8_000_000)
            return json.loads(body.decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as e:
            msg = _redact(f"HTTP {e.code}: {e.read(300).decode('utf-8', 'replace')}", key)
            last = LLMError("http_%d" % e.code, msg)
            if e.code not in (429, 500, 502, 503, 504):
                raise last
        except json.JSONDecodeError as e:
            raise LLMError("bad_json_envelope", _redact(str(e), key))
        except (TimeoutError, urllib.error.URLError, OSError) as e:
            reason = getattr(e, "reason", e)
            kind = "timeout" if isinstance(e, TimeoutError) or "timed out" in str(reason).lower() else "network"
            last = LLMError(kind, _redact(str(reason), key))
        if attempt < retries:
            time.sleep(backoff * (attempt + 1))
    raise last


class LLM:
    def __init__(self, provider=None, model=None, key=None, base_url=None, timeout=None, backoff=1.0):
        self.provider = (provider or os.getenv("PROMO_LLM_PROVIDER") or "none").lower()
        self.model = model or os.getenv("PROMO_LLM_MODEL") or {
            "gemini": "gemini-2.0-flash", "openai": "gpt-4o-mini", "anthropic": "claude-haiku-4-5"}.get(self.provider, "")
        self.base_url = base_url or os.getenv("PROMO_LLM_BASE_URL") or ""
        envkey = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(self.provider, "")
        self.key = key or os.getenv("PROMO_LLM_API_KEY") or (os.getenv(envkey) if envkey else "") or ""
        self.timeout = float(timeout or os.getenv("PROMO_LLM_TIMEOUT") or DEFAULT_TIMEOUT)
        self.backoff = backoff
        self.calls = 0

    @property
    def enabled(self):
        return self.provider in ("gemini", "openai", "anthropic") and (bool(self.key) or bool(self.base_url))

    def describe(self) -> dict:
        """Config that is safe to write to disk (never the key)."""
        host = urllib.parse.urlparse(self.base_url).netloc if self.base_url else "(provider default)"
        return {"provider": self.provider, "model": self.model, "endpoint_host": host, "timeout_s": self.timeout,
                "max_retries": MAX_RETRIES, "api_key_present": bool(self.key)}

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        t, b, k = self.timeout, self.backoff, self.key
        if self.provider == "gemini":
            base = self.base_url or "https://generativelanguage.googleapis.com"
            out = _post(f"{base}/v1beta/models/{self.model}:generateContent", {"x-goog-api-key": k}, {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}}, t, backoff=b, key=k)
            try:
                return out["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError):
                raise LLMError("bad_envelope", "unexpected Gemini response shape")
        if self.provider == "anthropic":
            base = self.base_url or "https://api.anthropic.com"
            out = _post(f"{base}/v1/messages", {"x-api-key": k, "anthropic-version": "2023-06-01"}, {
                "model": self.model, "max_tokens": 4096, "temperature": 0.2, "system": system,
                "messages": [{"role": "user", "content": user}]}, t, backoff=b, key=k)
            try:
                return "".join(blk.get("text", "") for blk in out["content"])
            except (KeyError, TypeError, AttributeError):
                raise LLMError("bad_envelope", "unexpected Anthropic response shape")
        if self.provider == "openai":
            base = (self.base_url or "https://api.openai.com/v1").rstrip("/")
            hdr = {"Authorization": f"Bearer {k}"} if k else {}
            out = _post(f"{base}/chat/completions", hdr, {
                "model": self.model, "temperature": 0.2, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}, t, backoff=b, key=k)
            try:
                return out["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                raise LLMError("bad_envelope", "unexpected OpenAI-compatible response shape")
        raise LLMError("unknown_provider", f"unknown provider {self.provider!r}")


# ---------------------------------------------------------------------------------------------------------
def _timeline(ep: Episode) -> str:
    rows = []
    for e in ep.events:
        if e.kind == "camera":
            continue
        who = e.speaker or e.kind.upper()
        tone = f" ({e.tone})" if e.tone else ""
        rows.append(f"{fmt(e.start)} {who}{tone}: {e.text[:220]}")
    return "\n".join(rows)


def build_prompt(ep: Episode, cands: list, title: str = "") -> str:
    lines = []
    for cid, c in cands:
        segs = ", ".join(f"[{fmt(s.start)}-{fmt(s.end)}]" for s in c.segments)
        lines.append(f'- id "{cid}": {segs} ({c.dur:.0f}s)')
    return PROMPT.format(title=title or "(untitled microdrama)", ep=ep.num, logline=ep.logline,
                         continuity=ep.continuity, timeline=_timeline(ep), cands="\n".join(lines))


def parse_ts(s) -> float | None:
    m = re.match(r"^\s*(\d+):(\d{1,2}(?:\.\d+)?)\s*$", str(s if s is not None else ""))
    if not m or float(m.group(2)) >= 60:
        return None
    return int(m.group(1)) * 60 + float(m.group(2))


def _norm(t: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", t.lower()).split())


def _num(v, lo, hi):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not (lo <= v <= hi):
        raise ValueError(f"not a finite number in [{lo}, {hi}]: {v!r}")
    return float(v)


def parse_response(text: str, expected_ids: set):
    """-> (items {id: clean dict}, rejects [(id|None, reason)]). Never raises on bad model output."""
    rejects: list = []
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.S).strip()
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}, [(None, "response is not valid JSON")]
    if not isinstance(obj, dict) or not isinstance(obj.get("results"), list):
        return {}, [(None, "response must be an object with a 'results' list")]
    seen: dict = {}
    for r in obj["results"]:
        rid = r.get("id") if isinstance(r, dict) else None
        if not isinstance(rid, str):
            rejects.append((None, "result without a string id"))
            continue
        seen.setdefault(rid, []).append(r)
    items = {}
    for rid, rs in seen.items():
        if rid not in expected_ids:
            rejects.append((rid, "unknown candidate id"))
            continue
        if len(rs) > 1:
            rejects.append((rid, "duplicate id in response"))
            continue
        r = rs[0]
        try:
            item = {"hook": _num(r.get("hook"), 1, 5), "cliffhanger": _num(r.get("cliffhanger"), 1, 5),
                    "standalone": _num(r.get("standalone"), 1, 5)}
            keep = r.get("keep", None)
            if keep is not None and not isinstance(keep, bool):
                raise ValueError("keep must be true/false")
            item["keep"] = keep
            for f in ("better_start", "better_end"):
                if r.get(f) is None:
                    item[f] = None
                else:
                    v = parse_ts(r[f])
                    if v is None:
                        raise ValueError(f"{f} is not mm:ss.xx")
                    item[f] = v
            for f, limit in (("hook_text", 200), ("cliffhanger_text", 200), ("why", 400)):
                v = r.get(f)
                if v is None or v == "":
                    item[f] = None
                elif isinstance(v, str) and len(v) <= limit:
                    item[f] = "".join(ch for ch in v if ch.isprintable()).strip()
                else:
                    raise ValueError(f"{f} must be a string of at most {limit} characters")
            items[rid] = item
        except ValueError as exc:
            rejects.append((rid, str(exc)))
    return items, rejects


def evidence_ok(ep: Episode, segments, hook_text, cliff_text) -> str | None:
    """Evidence must be copied (verbatim, ignoring case/punctuation) from script events that START inside the
    clip: hook -> first 4 s of the first segment, cliffhanger -> last 8 s of the last segment.
    Returns None if fine, else the reason."""
    def pool(seg, lo, hi):
        return " | ".join(_norm(e.text) for e in ep.events
                          if e.kind in ("dialogue", "action", "sfx") and e.start >= seg.start - 0.01 and e.start < seg.end
                          and e.end > lo and e.start < hi)
    if hook_text:
        s0 = segments[0]
        n = _norm(hook_text)
        if len(n) < 3 or n not in pool(s0, s0.start, s0.start + 4.0):
            return "hook_text is not quoted from the first seconds of the clip"
    if cliff_text:
        s1 = segments[-1]
        n = _norm(cliff_text)
        if len(n) < 3 or n not in pool(s1, s1.end - 8.0, s1.end):
            return "cliffhanger_text is not quoted from the end of the clip"
    return None
