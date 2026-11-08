import json
import os
import re
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from _common import CLEAN_ENV, episode_from  # noqa: E402
from promofinder import scoring as S
from promofinder.llm import LLM, evidence_ok, parse_response
from promofinder.pipeline import _trim, run
from promofinder.synthetic import write_demo_pack

IDS = {"E1-1", "E1-2"}


def item(i="E1-1", **kw):
    d = {"id": i, "hook": 4, "cliffhanger": 3, "standalone": 5, "keep": True}
    d.update(kw)
    return d


def resp(*items):
    return json.dumps({"results": list(items)})


class ResponseValidation(unittest.TestCase):
    def rejected(self, text):
        items, rej = parse_response(text, IDS)
        return items, [r[1] for r in rej]

    def test_valid_and_fenced_json(self):
        items, rej = parse_response("```json\n" + resp(item()) + "\n```", IDS)
        self.assertEqual(list(items), ["E1-1"])
        self.assertEqual(rej, [])

    def test_structure_errors(self):
        for bad in ["not json", "[]", '{"results": "x"}', "{}", ""]:
            items, rej = self.rejected(bad)
            self.assertEqual(items, {}, bad)
            self.assertTrue(rej, bad)

    def test_ids(self):
        items, rej = self.rejected(resp(item("E9-9"), item("E1-1"), item("E1-2"), item("E1-2")))
        self.assertEqual(list(items), ["E1-1"])
        self.assertTrue(any("unknown" in r for r in rej) and any("duplicate" in r for r in rej))
        items, rej = self.rejected(json.dumps({"results": [{"hook": 3}]}))
        self.assertEqual(items, {})

    def test_scores_must_be_finite_numbers_in_range(self):
        for bad in (0, 6, -1, True, "4", None, float("nan"), float("inf")):
            items, rej = self.rejected(json.dumps({"results": [item(hook=bad)]}))   # json.dumps emits NaN/Infinity
            self.assertEqual(items, {}, repr(bad))
        items, _ = self.rejected(resp(item(hook=1, cliffhanger=5, standalone=2.5)))
        self.assertEqual(len(items), 1)

    def test_other_fields(self):
        for kw in ({"keep": "yes"}, {"better_start": "soon"}, {"better_end": "99:99"}, {"why": "x" * 401}, {"hook_text": 5}):
            items, _ = self.rejected(resp(item(**kw)))
            self.assertEqual(items, {}, kw)
        items, _ = self.rejected(resp(item(better_start="00:12.50", why="ok")))
        self.assertEqual(items["E1-1"]["better_start"], 12.5)


class Evidence(unittest.TestCase):
    def setUp(self):
        ev = [("ACTION", 0.0, 1.0, "Start."), ("MARA", 1.0, 3.0, "You lied to the whole harbour!"),
              ("TOBIN", 25.0, 27.0, "Nobody. Yet."), ("ACTION", 30.0, 31.0, "End."), ("MARA", 50.0, 52.0, "Outside the clip entirely.")]
        self.ep = episode_from(ev, scene_end=55.0)
        self.segs = [S.Segment(1, 1.0, 31.0)]

    def test_verbatim_quotes_inside_clip_pass(self):
        self.assertIsNone(evidence_ok(self.ep, self.segs, "you lied to the whole harbour", "Nobody. Yet."))

    def test_fabricated_or_out_of_clip_evidence_is_rejected(self):
        self.assertIsNotNone(evidence_ok(self.ep, self.segs, "She screamed his name", None))
        self.assertIsNotNone(evidence_ok(self.ep, self.segs, None, "Outside the clip entirely"))
        self.assertIsNotNone(evidence_ok(self.ep, self.segs, "Nobody. Yet.", None))   # real line, but not in the first seconds


class Trims(unittest.TestCase):
    def setUp(self):
        ev = [("ACTION", 0.0, 1.0, "s"), ("A", 1.0, 3.0, "one"), ("A", 5.0, 9.0, "two long line"), ("A", 12.0, 14.0, "three"),
              ("A", 26.0, 28.0, "four"), ("A", 40.0, 44.0, "five long line"), ("ACTION", 45.0, 46.0, "e")]
        self.ep = episode_from(ev, scene_end=50.0)
        self.c = S.score_candidate(self.ep, S.Candidate([S.Segment(1, 1.0, 28.0)]))

    def test_valid_trim_is_applied(self):
        c2, why = _trim(self.ep, self.c, 5.0, None)
        self.assertIsNone(why)
        self.assertEqual((c2.segments[0].start, c2.segments[0].end), (5.0, 28.0))

    def test_trim_into_dialogue_snaps_to_a_legal_point_or_is_rejected(self):
        c2, why = _trim(self.ep, self.c, 7.0, None)             # 7.0 is inside 5-9; nearest legal start is 5.0 or 12.0
        for s in c2.segments:
            self.assertFalse(5.0 < s.start < 9.0)

    def test_trim_that_breaks_duration_is_rejected_and_original_kept(self):
        c2, why = _trim(self.ep, self.c, 12.0, None)           # 12 -> 28 = 16 s < 20
        self.assertIs(c2, self.c)
        self.assertIn("duration_out_of_range", why)

    def test_trim_far_from_any_cut_point_is_rejected(self):
        c2, why = _trim(self.ep, self.c, None, 36.0)
        self.assertIs(c2, self.c)
        self.assertIn("legal cut point", why)


class Handler(BaseHTTPRequestHandler):
    mode = "good"
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
        ids = sorted(set(re.findall(r"E\d+-\d+", body)))
        m = Handler.mode
        if m == "slow":
            time.sleep(2.0)
        if m == "500":
            self.send_response(500); self.end_headers(); self.wfile.write(b"boom"); return
        if m == "fabricated":
            res = {"results": [item(i, hook_text="a line that is not in the script") for i in ids]}
        elif m == "garbage":
            res = "this is not json"
        else:
            res = {"results": [item(i) for i in ids]}
        s = res if isinstance(res, str) else json.dumps(res)
        path = self.path
        if "generateContent" in path:
            out = {"candidates": [{"content": {"parts": [{"text": s}]}}]}
        elif path.endswith("/messages"):
            out = {"content": [{"type": "text", "text": s}]}
        else:
            out = {"choices": [{"message": {"content": s}}]}
        b = json.dumps(out).encode()
        try:
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        except (BrokenPipeError, ConnectionResetError):
            pass


class ProviderTests(unittest.TestCase):
    """Local fake provider on 127.0.0.1 only; no credentials, no internet. Every request has a short timeout."""

    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.srv.daemon_threads = True
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"
        cls.pack = write_demo_pack(Path(tempfile.mkdtemp()) / "eps", 3)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def go(self, mode, provider="openai", timeout=1.0):
        Handler.mode = mode
        base = self.base + ("/v1" if provider == "openai" else "")
        llm = LLM(provider, "stub", "TEST-KEY-123456789", base, timeout=timeout, backoff=0.0)
        t0 = time.time()
        with mock.patch.dict(os.environ, CLEAN_ENV, clear=True):
            recs, summ = run(self.pack, Path(tempfile.mkdtemp()), llm=llm, log=lambda *_: None)
        self.assertLess(time.time() - t0, 30, "run must terminate promptly")
        return recs, summ

    def test_all_three_provider_formats(self):
        for prov in ("gemini", "anthropic", "openai"):
            with self.subTest(prov):
                recs, summ = self.go("good", prov)
                self.assertEqual(summ["llm"]["episodes_ok"], 3)
                self.assertEqual(summ["llm"]["episodes_fallback_to_rules"], 0)
                self.assertTrue(all(r["mode"] == "llm+rules" for r in recs))
                self.assertEqual(summ["validation"]["violations"], 0)

    def test_http_error_falls_back_visibly(self):
        recs, summ = self.go("500")
        L = summ["llm"]
        self.assertEqual(L["episodes_fallback_to_rules"], 3)
        self.assertEqual({f["kind"] for f in L["failures"]}, {"http_500"})
        self.assertTrue(recs and all(r["mode"] == "rules" for r in recs))

    def test_invalid_json_falls_back_visibly(self):
        recs, summ = self.go("garbage")
        self.assertEqual(summ["llm"]["episodes_fallback_to_rules"], 3)
        self.assertTrue(summ["llm"]["items_rejected"])

    def test_timeout_is_bounded(self):
        recs, summ = self.go("slow", timeout=0.3)
        self.assertEqual({f["kind"] for f in summ["llm"]["failures"]}, {"timeout"})

    def test_fabricated_evidence_rejected(self):
        recs, summ = self.go("fabricated")
        L = summ["llm"]
        self.assertEqual(L["items_accepted"], 0)
        self.assertTrue(any("not quoted" in x["reason"] for x in L["items_rejected"]))
        self.assertTrue(all(r["mode"] == "rules" for r in recs))

    def test_loopback_ignores_proxy_environment(self):
        env = dict(CLEAN_ENV, HTTP_PROXY="http://10.255.255.1:3128", http_proxy="http://10.255.255.1:3128")
        env.pop("NO_PROXY", None); env.pop("no_proxy", None)
        Handler.mode = "good"
        llm = LLM("openai", "stub", "k", self.base + "/v1", timeout=1.0, backoff=0.0)
        t0 = time.time()
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIn("results", llm.complete("s", "E1-1"))
        self.assertLess(time.time() - t0, 5)

    def test_key_never_written_to_outputs(self):
        _, summ = self.go("500")
        out = json.dumps(summ)
        self.assertNotIn("TEST-KEY-123456789", out)
        self.assertTrue(summ["llm"]["config"]["api_key_present"])


class NoCredentials(unittest.TestCase):
    def test_requested_without_key_is_rules_only_and_recorded(self):
        pack = write_demo_pack(Path(tempfile.mkdtemp()) / "eps", 2)
        with mock.patch.dict(os.environ, CLEAN_ENV, clear=True):
            _, summ = run(pack, Path(tempfile.mkdtemp()), llm=LLM("gemini"), log=lambda *_: None)
        self.assertFalse(summ["llm"]["used"])
        self.assertIn("not_used_reason", summ["llm"])


if __name__ == "__main__":
    unittest.main()
