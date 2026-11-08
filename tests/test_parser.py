import tempfile
import unittest
from pathlib import Path

from _common import write  # noqa: E402
from promofinder.parser import load_pack, parse_episode
from promofinder.pipeline import PackError, run
from promofinder.synthetic import demo_episode, episode_text

GOOD = [("MARA", 1.0, 3.0, "Hello there."), ("CAMERA", 1.0, 4.0, "Wide shot."), ("ACTION", 2.0, 5.0, "A door opens.")]


def codes(ep):
    return [d.code for d in ep.diagnostics]


class ParserDiagnostics(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_overlapping_tracks_are_kept_untouched(self):
        ep = parse_episode(write(self.d, "e1.txt", episode_text(1, GOOD)))
        self.assertEqual(len(ep.events), 3)
        self.assertEqual(sorted((e.kind, e.start, e.end) for e in ep.events),
                         [("action", 2.0, 5.0), ("camera", 1.0, 4.0), ("dialogue", 1.0, 3.0)])
        self.assertEqual(ep.n_errors, 0)

    def test_reversed_timing_is_reported_not_repaired(self):
        text = episode_text(1, GOOD).replace("Start: (00:01.00) | End: (00:03.00)", "Start: (00:09.00) | End: (00:03.00)")
        ep = parse_episode(write(self.d, "e1.txt", text))
        self.assertIn("reversed_timing", codes(ep))
        self.assertEqual(len(ep.events), 2)                       # the bad event is excluded ...
        self.assertEqual(len(ep.invalid_events), 1)               # ... and listed
        self.assertFalse(any(e.end == e.start for e in ep.events))  # no zero-length repair

    def test_missing_malformed_and_out_of_range_timestamps(self):
        t = episode_text(1, GOOD)
        for bad, code in [("Start: (00:01.00) | End: (00:03.00)", "malformed_timing"),
                          ("Start: (00:01.00) | End: (00:75.00)", "invalid_timestamp")]:
            repl = "Start: (soon) | End: (later)" if code == "malformed_timing" else bad
            text = t.replace("Start: (00:01.00) | End: (00:03.00)", repl)
            ep = parse_episode(write(self.d, f"{code}.txt", text))
            self.assertIn(code, codes(ep))
        missing = t.replace("Start: (00:01.00) | End: (00:03.00)\n", "")
        ep = parse_episode(write(self.d, "missing.txt", missing))
        self.assertIn("missing_timing", codes(ep))

    def test_empty_and_headerless_and_bad_encoding(self):
        ep = parse_episode(write(self.d, "episode-007-empty.txt", ""))
        self.assertIn("no_events", codes(ep))
        ep = parse_episode(write(self.d, "episode-008-x.txt", episode_text(8, GOOD, header=False)))
        self.assertIn("episode_number_from_filename", codes(ep))
        self.assertEqual(ep.num, 8)
        p = Path(self.d) / "episode-009-bin.txt"
        p.write_bytes(b"\xff\xfe\x00bad")
        self.assertIn("bad_encoding", codes(parse_episode(p)))
        ep = parse_episode(write(self.d, "noname.txt", episode_text(1, GOOD, header=False)))
        self.assertIn("no_episode_number", codes(ep))

    def test_duplicate_episode_numbers_are_errors_and_excluded(self):
        write(self.d, "a.txt", episode_text(3, GOOD))
        write(self.d, "b.txt", episode_text(3, GOOD))
        write(self.d, "c.txt", episode_text(4, GOOD))
        res = load_pack(self.d)
        self.assertIn("duplicate_episode", [x.code for x in res.errors])
        self.assertEqual([e.num for e in res.episodes], [4])

    def test_missing_episode_numbers_reported(self):
        for n in (1, 2, 5):
            write(self.d, f"e{n}.txt", episode_text(n, GOOD))
        res = load_pack(self.d)
        miss = [x for x in res.diagnostics if x.code == "missing_episode_numbers"]
        self.assertEqual(len(miss), 1)
        self.assertIn("[3, 4]", miss[0].message)

    def test_missing_input_and_empty_folder(self):
        self.assertIn("input_not_found", [x.code for x in load_pack(Path(self.d) / "nope").errors])
        self.assertIn("no_input_files", [x.code for x in load_pack(self.d).errors])


class RunFailsLoudly(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        write(self.d, "episode-001-ok.txt", demo_episode(1, 90.0))
        bad = episode_text(2, GOOD).replace("Start: (00:01.00) | End: (00:03.00)", "Start: (00:09.00) | End: (00:03.00)")
        write(self.d, "episode-002-bad.txt", bad)

    def test_default_stops_and_writes_diagnostics(self):
        out = self.d / "out"
        with self.assertRaises(PackError) as cm:
            run(self.d, out, log=lambda *_: None)
        self.assertIn("reversed_timing", str(cm.exception))
        self.assertTrue((out / "diagnostics.json").exists())
        self.assertFalse((out / "promos.json").exists())

    def test_continue_on_error_processes_valid_episodes_and_lists_the_rest(self):
        out = self.d / "out2"
        recs, summ = run(self.d, out, continue_on_error=True, log=lambda *_: None)
        self.assertEqual({r["episode"] for r in recs}, {1})
        self.assertEqual(summ["episodes_loaded"], 1)
        self.assertEqual(summ["files_seen"], 2)
        self.assertGreaterEqual(summ["diagnostics"]["errors"], 1)
        self.assertIn("warning:episode_skipped", summ["diagnostics"]["by_code"])


if __name__ == "__main__":
    unittest.main()
