# promofinder - find promo clips in microdrama screenplays

Reads timestamped screenplays (one `.txt` per episode) and returns ranked **promos**: 20-60 s clips that open on a
hook, end on a cliffhanger and make sense to someone who has not seen the show. Each promo lists the episode, the
start/end of every segment, hook, cliffhanger, a one-line reason, scores and risk flags.

Python **3.8+** (static check with `vermin`; the test-suite was run on 3.11, 3.12 and 3.13 only). No third-party packages.

## Try it without any data (synthetic demo)
macOS / Linux:
```bash
python3 -m promofinder.demo demo_out
python3 -m unittest discover -s tests -v
```
Windows PowerShell:
```powershell
py -3 -m promofinder.demo demo_out
py -3 -m unittest discover -s tests -v
```
Open `demo_out/promos.html`.

## Run on a folder of screenplays
```powershell
py -3 -m promofinder "C:\path\to\episodes" -o out --title "Show name"
```
```bash
python3 -m promofinder /path/to/episodes -o out --title "Show name"
```
Outputs in `out/`: `promos.html` (review page), `promos.csv`, `promos.json`, `promos_ranked.md`,
`diagnostics.json` (every input problem), `run_summary.json`, `run_manifest.json` (config + SHA-256 of code and inputs).

If any file has an error (reversed or malformed timestamp, missing timing, duplicate episode number, ...) the run
stops, writes `diagnostics.json` and exits with code 2. `--continue-on-error` processes only the valid episodes and
keeps the problems listed. Nothing is repaired or guessed.

## Cut-point rules (one shared validator, `promofinder/validate.py`)
Every exported segment is checked, whether it came from the rule engine, a stitched cut or an LLM trim:
1. finite `start < end`, inside the episode (0 .. episode duration);
2. start = the start of a dialogue/action/sfx event (or scene start); end = the end of such an event (or scene end);
3. **no boundary strictly inside a spoken-dialogue interval**;
4. segments in chronological order, not overlapping;
5. total duration 20-60 s.

Action, camera and sfx intervals may overlap dialogue in these scripts, so a cut **may cross** them (their start/end times are
legal cut points). Only spoken lines are protected; there is no picture to check that an action beat is visually complete.

## Optional LLM judge
The rules propose candidates; a model can re-score the best 5 per episode. Provider-agnostic, standard library only, per-request
timeout (default 30 s, `--llm-timeout`), at most 2 retries. Key from the environment only (never written to disk).
Everything the model returns is validated: JSON structure, known ids, finite scores in range, evidence quoted verbatim from
inside the clip, and every trim re-checked by the shared validator. Rejected items, trims and provider failures (with the rule
score kept as fallback) are listed in `run_summary.json` under `llm`.
```powershell
$env:GEMINI_API_KEY = "<your key>"
py -3 -m promofinder "C:\path\to\episodes" -o out_llm --llm gemini --model gemini-2.0-flash
```
```bash
export GEMINI_API_KEY=...   # or --llm anthropic / --llm openai [--base-url http://localhost:11434/v1]
```
**Status:** the LLM stage is tested only against a local fake provider (`tests/test_llm.py`, 127.0.0.1 only, no credentials).
It has not been run against a live provider.

## Scaling
Let N = episodes, E = events in an episode, w = events inside one 60 s window. The parser is O(N*E). Candidate generation is
O(N * E * w) windows, each scored in O(w), so the rule stage is roughly O(N * E * w^2) and is linear in N. Cost grows faster in
event density than in episode count. `tools/bench_scaling.py` measures it on synthetic data.

## Confidentiality
No Kuku screenplay text is in this repository. Tests use invented synthetic scripts. The public ZIP is built from an explicit
allowlist: `python tools/build_public_zip.py` (it lists and verifies its own contents).

## Known limits
Screenplay text only (no picture/audio); English word lists; no cross-episode promos; rule scores are heuristics (see the solution doc for the evaluation status).
