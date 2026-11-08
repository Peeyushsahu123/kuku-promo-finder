"""End-to-end: parse -> validate -> candidates -> (optional, validated LLM judge) -> rank -> reports + evidence."""
from __future__ import annotations
import csv
import hashlib
import html
import json
import platform
import string
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from . import scoring as S
from .llm import (LLM, LLMError, SYSTEM, build_prompt, evidence_ok, parse_response)
from .parser import Episode, fmt, load_pack
from .validate import build_indexes, validate_segments

EARLY_W = 0.30          # post-first bonus for early episodes (a clicking viewer starts at ep 1)
EARLY_SPAN = 30
MAX_OVERLAP = 0.25
TRIM_SNAP_SECONDS = 2.0


class PackError(Exception):
    """Input problems that stop the run (diagnostics.json is written first)."""


# ---------- human-readable text from the script (rule mode) ----------------------------------------
def _beats(ep: Episode, a: float, b: float):
    return [e for e in S._window_events(ep, a, b) if e.kind in ("dialogue", "action", "sfx")]


def _line(e) -> str:
    if e.kind == "dialogue":
        who = string.capwords(e.speaker)
        return f'{who}: "{e.text}"'
    return e.text


def hook_text(ep: Episode, cand: S.Candidate) -> str:
    a = cand.segments[0].start
    evs = [e for e in S._speech_events(ep) if e.end > a and e.start < a + S.HOOK_SECS]
    return " / ".join(_line(e)[:140] for e in evs[:2]) or "(no event in first 3 s)"


def cliff_text(ep: Episode, cand: S.Candidate) -> str:
    b = cand.segments[-1].end
    evs = [e for e in S._speech_events(ep) if e.end > b - 7 and e.start < b + 0.05 and e.kind != "sfx"]
    return _line(evs[-1])[:160] if evs else "(none)"


def why_text(ep: Episode, cand: S.Candidate) -> str:
    p = cand.parts
    bits = []
    if p["hook"] >= 0.7:
        bits.append("opens on a charged line/beat inside 3 s")
    elif p["hook"] >= 0.5:
        bits.append("opens with some tension")
    b = cand.segments[-1].end
    tail = [e for e in S._speech_events(ep) if e.end > b - 7 and e.start < b + 0.05 and e.kind != "sfx"]
    if tail:
        last = tail[-1]
        if last.kind == "dialogue" and last.text.rstrip().endswith("?"):
            bits.append("cuts on an unanswered question")
        elif S.RE_REVEAL.search(last.text):
            bits.append("cuts on a reveal/threat")
        elif p["cliff"] >= 0.65:
            bits.append("cuts at a high-tension beat")
    if p["esc"] >= 0.7:
        bits.append("tension escalates through the clip")
    if p["alone"] >= 0.7:
        bits.append("2-3 characters + premise words make it followable cold")
    if len(cand.segments) > 1:
        bits.append("two-part cut: peak moment + closing beat (skips the middle)")
    return "; ".join(bits).capitalize() + "." if bits else "Weak signals across the board."


def risk_flags(ep: Episode, cand: S.Candidate) -> list[str]:
    p, f = cand.parts, []
    if p["hook"] < 0.5:
        f.append("weak-hook")
    if p["cliff"] < 0.5:
        f.append("weak-cliffhanger")
    if p["alone"] < 0.55:
        f.append("needs-context")
    if ep.num > 25:
        f.append("late-episode")
    if len(cand.segments) > 1:
        f.append("stitched")
    a = cand.segments[0].start
    first = next((e for e in S._speech_events(ep) if e.end > a), None)
    if first and first.start < a - 1.0:
        f.append("starts-mid-beat")
    if p["pace"] < 0.6:
        f.append("slow-stretch")
    return f


# ---------- main -----------------------------------------------------------------------------------
def _record(ep: Episode, cand: S.Candidate, mode: str, extra: dict | None = None) -> dict:
    r = {
        "episode": ep.num,
        "segments": [{"episode": s.ep, "start": fmt(s.start), "end": fmt(s.end), "start_s": round(s.start, 2),
                      "end_s": round(s.end, 2)} for s in cand.segments],
        "duration_s": round(cand.dur, 2),
        "hook": hook_text(ep, cand),
        "cliffhanger": cliff_text(ep, cand),
        "why": why_text(ep, cand),
        "scores": {k: round(v, 3) for k, v in cand.parts.items()},
        "rule_score": round(cand.score, 4),
        "flags": risk_flags(ep, cand),
        "mode": mode,
        "transcript": [f"{fmt(e.start)} {_line(e)}" for s in cand.segments for e in _beats(ep, s.start, s.end)],
    }
    if extra:
        r.update(extra)
    return r


def _trim(ep: Episode, cand: S.Candidate, st, en):
    """Apply an LLM-suggested trim. Returns (candidate, reason_if_rejected). The trim is snapped to LEGAL cut
    points (not inside dialogue) and then re-checked by the shared validator; if that fails the original stays."""
    if st is None and en is None:
        return cand, None
    if len(cand.segments) != 1:
        return cand, "trim ignored for stitched promos"
    starts, ends = S._anchors(ep)
    seg = cand.segments[0]
    a, b = seg.start, seg.end
    if st is not None:
        near = min(starts, key=lambda x: abs(x - st))
        if abs(near - st) > TRIM_SNAP_SECONDS:
            return cand, f"better_start {st:.2f} is not within {TRIM_SNAP_SECONDS}s of a legal cut point"
        a = near
    if en is not None:
        near = min(ends, key=lambda x: abs(x - en))
        if abs(near - en) > TRIM_SNAP_SECONDS:
            return cand, f"better_end {en:.2f} is not within {TRIM_SNAP_SECONDS}s of a legal cut point"
        b = near
    if (a, b) == (seg.start, seg.end):
        return cand, None
    viol = validate_segments({ep.num: S.get_index(ep)}, [(ep.num, a, b)])
    if viol:
        return cand, "; ".join(str(v) for v in viol)
    return S.score_candidate(ep, S.Candidate([S.Segment(ep.num, a, b)])), None


def _llm_stage(llm: LLM, ep: Episode, cands: list, per_episode: int, title: str, stats: dict, log):
    """Returns (chosen [(final_score, cand, item|None)], used_llm_for_this_episode)."""
    ids = [(f"E{ep.num}-{i + 1}", c) for i, c in enumerate(cands)]
    stats["episodes_attempted"] += 1
    stats["items_requested"] += len(ids)
    try:
        text = llm.complete(SYSTEM, build_prompt(ep, ids, title))
    except LLMError as exc:
        stats["episodes_fallback_to_rules"] += 1
        stats["failures"].append({"episode": ep.num, "kind": exc.kind, "message": str(exc)[:300]})
        log(f"  ep{ep.num}: provider failure ({exc.kind}); using rule scores for this episode")
        return [(c.score, c, None) for c in cands[:per_episode]], False
    items, rejects = parse_response(text, {cid for cid, _ in ids})
    for rid, reason in rejects:
        stats["items_rejected"].append({"episode": ep.num, "id": rid, "reason": reason})
    if not items:
        stats["episodes_fallback_to_rules"] += 1
        stats["failures"].append({"episode": ep.num, "kind": "no_valid_items", "message": "no usable results in response"})
        return [(c.score, c, None) for c in cands[:per_episode]], False
    stats["episodes_ok"] += 1
    rows = []
    for cid, c in ids:
        it = items.get(cid)
        c2, used = c, None
        if it:
            c_trim, why_not = _trim(ep, c, it["better_start"], it["better_end"])
            wanted_trim = it["better_start"] is not None or it["better_end"] is not None
            ev = evidence_ok(ep, c_trim.segments, it["hook_text"], it["cliffhanger_text"])
            if ev:
                stats["items_rejected"].append({"episode": ep.num, "id": cid, "reason": ev})
            else:
                c2, used = c_trim, it
                if why_not:
                    stats["trims_rejected"].append({"episode": ep.num, "id": cid, "reason": why_not})
                elif wanted_trim and c_trim is not c:
                    stats["trims_applied"] += 1
        if used:
            stats["items_accepted"] += 1
            ln = (0.35 * used["hook"] + 0.35 * used["cliffhanger"] + 0.30 * used["standalone"]) / 5.0
            if used["keep"] is False:
                ln *= 0.6
            final = 0.4 * c2.score + 0.6 * ln * S.duration_factor(c2.dur)
        else:
            final = c.score
        rows.append((final, c2, used))
    rows.sort(key=lambda x: -x[0])
    chosen = []
    for row in rows:
        if all(S.overlap(row[1], x[1]) <= MAX_OVERLAP for x in chosen):
            chosen.append(row)
        if len(chosen) >= per_episode:
            break
    return chosen, True


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest(input_path, files, args: dict, llm: LLM | None) -> dict:
    pkg = Path(__file__).parent
    code = {p.name: _sha256(p) for p in sorted(pkg.glob("*.py"))}
    combined = hashlib.sha256("".join(f"{k}:{v}" for k, v in sorted(code.items())).encode()).hexdigest()
    return {
        "tool": "promofinder", "version": __version__, "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "config": {**args, "constants": {
            "weights": S.WEIGHTS, "min_len": S.MIN_LEN, "max_len": S.MAX_LEN, "ideal": [S.IDEAL_LO, S.IDEAL_HI],
            "hook_secs": S.HOOK_SECS, "early_bonus_weight": EARLY_W, "early_span_eps": EARLY_SPAN,
            "max_overlap": MAX_OVERLAP, "trim_snap_s": TRIM_SNAP_SECONDS},
            "llm": llm.describe() if llm else {"provider": "none"}},
        "code_sha256": code, "code_combined_sha256": combined,
        "input_files_sha256": {Path(f).name: _sha256(f) for f in files},
    }


def run(input_path, out_dir, per_episode=2, stitched=True, top=20, llm: LLM | None = None,
        llm_pool=5, title="", continue_on_error=False, log=print):
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    loaded = load_pack(input_path)
    diags = [d.as_dict() for d in loaded.diagnostics]
    (out / "diagnostics.json").write_text(json.dumps(diags, indent=2, ensure_ascii=False))
    errors = [d for d in diags if d["severity"] == "error"]
    if errors and not continue_on_error:
        msg = "\n".join(f"  [{d['code']}] {d['file']}:{d['line'] or '-'} {d['message']}" for d in errors[:15])
        raise PackError(f"{len(errors)} input error(s); nothing was processed. See {out / 'diagnostics.json'}.\n{msg}\n"
                        "Fix the files, or pass --continue-on-error to process only the valid episodes.")
    eps = loaded.episodes
    if not eps:
        raise PackError(f"no usable episodes. See {out / 'diagnostics.json'}.")
    log(f"parsed {len(eps)} usable episodes ({loaded.files_seen} files), {sum(e.duration for e in eps) / 60:.1f} min, "
        f"{len(errors)} error / {sum(1 for d in diags if d['severity'] == 'warning')} warning diagnostics")
    indexes = build_indexes(eps)
    use_llm = bool(llm and llm.enabled)
    stats = {"episodes_attempted": 0, "episodes_ok": 0, "episodes_fallback_to_rules": 0, "failures": [],
             "items_requested": 0, "items_accepted": 0, "items_rejected": [], "trims_applied": 0, "trims_rejected": []}
    records, pool_sizes, no_promo = [], [], []
    for ep in eps:
        pool_n = per_episode if not use_llm else max(per_episode, llm_pool)
        cands, n = S.find_candidates(ep, per_ep=pool_n, stitched=stitched)
        pool_sizes.append(n)
        if not cands:
            no_promo.append({"episode": ep.num, "reason": "no valid 20-60 s window with legal cut points"})
            continue
        if use_llm:
            chosen, _ = _llm_stage(llm, ep, cands, per_episode, title, stats, log)
        else:
            chosen = [(c.score, c, None) for c in cands[:per_episode]]
        for final, c, item in chosen:
            c.score = final
            extra = None
            if item:
                extra = {"llm": {k: item[k] for k in ("hook", "cliffhanger", "standalone", "keep")},
                         "hook_text_llm_verified": item["hook_text"], "cliffhanger_text_llm_verified": item["cliffhanger_text"]}
            rec = _record(ep, c, "llm+rules" if item else "rules", extra)
            if item and item["why"]:
                rec["why_llm"] = item["why"]            # written by the model; not used as evidence
            rec["_score"] = c.score
            records.append(rec)

    for r in records:
        early = max(0.0, 1.0 - (r["episode"] - 1) / EARLY_SPAN)
        r["rank_score"] = round(r["_score"] * (1 + EARLY_W * early), 4)
    records.sort(key=lambda r: (-r["rank_score"], r["episode"], r["segments"][0]["start_s"]))
    for i, r in enumerate(records, 1):
        r["rank"] = i
        r["tier"] = "A - post first" if i <= top // 2 else ("B - post next" if i <= top * 2 else "C - backup")
        r["id"] = f"P{i:03d}"
        r.pop("_score", None)

    # final gate: EVERY exported segment must pass the shared validator
    bad, nseg = [], 0
    for r in records:
        segs = [(s["episode"], s["start_s"], s["end_s"]) for s in r["segments"]]
        nseg += len(segs)
        v = validate_segments(indexes, segs)
        r["validation"] = "passed" if not v else "FAILED"
        if v:
            bad.append({"id": r["id"], "violations": [str(x) for x in v]})
    if bad:
        (out / "validation_failures.json").write_text(json.dumps(bad, indent=2))
        raise RuntimeError(f"{len(bad)} promo(s) failed validation; see validation_failures.json (this is a bug)")

    (out / "promos.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    _write_csv(out / "promos.csv", records)
    _write_md(out / "promos_ranked.md", records, top)
    _write_html(out / "promos.html", records, title)
    llm_block = {"requested": llm is not None and llm.provider != "none", "used": use_llm, "calls": llm.calls if llm else 0}
    if llm is not None and llm.provider != "none":
        llm_block["config"] = llm.describe()
        if use_llm:
            llm_block.update(stats)
        else:
            llm_block["not_used_reason"] = "no API key or base URL available; rules only"
    summary = {
        "episodes_loaded": len(eps), "files_seen": loaded.files_seen, "minutes": round(sum(e.duration for e in eps) / 60, 1),
        "promos": len(records), "single_segment": sum(1 for r in records if len(r["segments"]) == 1),
        "stitched": sum(1 for r in records if len(r["segments"]) > 1),
        "episodes_with_promos": len({r["episode"] for r in records}), "episodes_without_promos": no_promo,
        "mean_duration_s": round(sum(r["duration_s"] for r in records) / max(1, len(records)), 1),
        "valid_candidates_scored_total": sum(pool_sizes),
        "validation": {"promos_checked": len(records), "segments_checked": nseg, "violations": 0, "rules": "V1-V5, see validate.py"},
        "diagnostics": {"errors": len(errors), "warnings": sum(1 for d in diags if d["severity"] == "warning"),
                        "info": sum(1 for d in diags if d["severity"] == "info"),
                        "by_code": _count_codes(diags)},
        "llm": llm_block, "flag_counts": _count_flags(records), "runtime_s": round(time.time() - t0, 1),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    args = {"per_episode": per_episode, "stitched": stitched, "top": top, "llm_pool": llm_pool, "title": title,
            "continue_on_error": continue_on_error}
    (out / "run_manifest.json").write_text(json.dumps(_manifest(
        input_path, sorted(Path(input_path).rglob("*.txt")) if Path(input_path).is_dir() else [Path(input_path)], args, llm), indent=2))
    log(json.dumps({k: summary[k] for k in ("episodes_loaded", "promos", "single_segment", "stitched", "validation", "runtime_s")}, indent=2))
    return records, summary


def _count_codes(diags):
    c = {}
    for d in diags:
        k = f"{d['severity']}:{d['code']}"
        c[k] = c.get(k, 0) + 1
    return dict(sorted(c.items()))


def _count_flags(records):
    c = {}
    for r in records:
        for f in r["flags"]:
            c[f] = c.get(f, 0) + 1
    return dict(sorted(c.items(), key=lambda x: -x[1]))


def _segs(r):
    return " + ".join(f"Ep{s['episode']} {s['start']}-{s['end']}" for s in r["segments"])


def _write_csv(path, records):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "id", "tier", "episode", "segments", "duration_s", "hook", "cliffhanger", "why",
                    "rank_score", "hook_score", "cliff_score", "standalone_score", "flags", "mode"])
        for r in records:
            s = r["scores"]
            w.writerow([r["rank"], r["id"], r["tier"], r["episode"], _segs(r), r["duration_s"], r["hook"],
                        r["cliffhanger"], r["why"], r["rank_score"], s["hook"], s["cliff"], s["alone"],
                        "|".join(r["flags"]), r["mode"]])


def _write_md(path, records, top):
    L = ["# Ranked promos\n",
         f"{len(records)} promos from {len({r['episode'] for r in records})} episodes. Tier A = post first.\n"]
    for r in records:
        L.append(f"## {r['rank']}. {r['id']}  ({r['tier']})  -  {_segs(r)}  [{r['duration_s']}s]")
        L.append(f"- **Hook (first 3 s):** {r['hook']}")
        L.append(f"- **Cliffhanger:** {r['cliffhanger']}")
        L.append(f"- **Why it works:** {r['why']}")
        s = r["scores"]
        L.append(f"- Scores: hook {s['hook']} / cliff {s['cliff']} / standalone {s['alone']} / escalation {s['esc']} / pace {s['pace']}"
                 f" -> rank score {r['rank_score']}" + (f"  | flags: {', '.join(r['flags'])}" if r["flags"] else ""))
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def _write_html(path, records, title):
    cards = []
    for r in records:
        tr = "<br>".join(html.escape(x) for x in r["transcript"])
        fl = " ".join(f'<span class="f">{html.escape(f)}</span>' for f in r["flags"])
        s = r["scores"]
        cards.append(f"""<article><h3><span class="rk">#{r['rank']}</span> {html.escape(_segs(r))} <small>{r['duration_s']}s · {html.escape(r['tier'])}</small></h3>
<p><b>Hook:</b> {html.escape(r['hook'])}</p><p><b>Cliffhanger:</b> {html.escape(r['cliffhanger'])}</p>
<p><b>Why:</b> {html.escape(r['why'])}</p>
<p class="m">hook {s['hook']} · cliff {s['cliff']} · standalone {s['alone']} · esc {s['esc']} · pace {s['pace']} · rank {r['rank_score']} {fl}</p>
<details><summary>Transcript</summary><p class="t">{tr}</p></details></article>""")
    path.write_text(f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Promo finder - {html.escape(title or 'results')}</title><style>
:root{{--bg:#fff;--fg:#1c1c22;--mut:#6b6b78;--card:#f5f5f8;--acc:#7a3cff}}
@media(prefers-color-scheme:dark){{:root{{--bg:#15151a;--fg:#ececf2;--mut:#9a9aa8;--card:#20202a;--acc:#a98bff}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:860px;margin:0 auto;padding:16px}}
article{{background:var(--card);border-radius:10px;padding:12px 16px;margin:12px 0}}h3{{margin:.2em 0}}h3 small{{color:var(--mut);font-weight:400}}
.rk{{color:var(--acc)}}.m{{color:var(--mut);font-size:13px}}.f{{background:var(--acc);color:#fff;border-radius:4px;padding:0 6px;font-size:12px;margin-left:4px}}
.t{{font-size:13px;color:var(--mut)}}p{{margin:.35em 0}}</style>
<h1>Ranked promos</h1><p class="m">{len(records)} promos · generated by promofinder</p>{''.join(cards)}""", encoding="utf-8")
