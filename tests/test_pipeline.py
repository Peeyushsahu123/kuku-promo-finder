import json
import tempfile
import unittest
from pathlib import Path

from _common import write  # noqa: E402
from promofinder.pipeline import run
from promofinder.synthetic import write_demo_pack, demo_episode


class PipelineEndToEnd(unittest.TestCase):
    def test_demo_is_valid_and_deterministic(self):
        pack = write_demo_pack(Path(tempfile.mkdtemp()) / "eps", 3)
        outs = []
        for _ in range(2):
            out = Path(tempfile.mkdtemp())
            recs, summ = run(pack, out, log=lambda *_: None)
            outs.append(json.dumps(recs, sort_keys=True))
            self.assertEqual(summ["validation"]["violations"], 0)
            self.assertTrue(all(r["validation"] == "passed" and 20 <= r["duration_s"] <= 60 for r in recs))
            for f in ("promos.json", "promos.csv", "promos.html", "promos_ranked.md", "diagnostics.json",
                      "run_summary.json", "run_manifest.json"):
                self.assertTrue((out / f).exists(), f)
        self.assertEqual(outs[0], outs[1])

    def test_manifest_has_hashes_and_no_secrets(self):
        pack = write_demo_pack(Path(tempfile.mkdtemp()) / "eps", 1)
        out = Path(tempfile.mkdtemp())
        run(pack, out, log=lambda *_: None)
        m = json.loads((out / "run_manifest.json").read_text())
        self.assertEqual(len(m["code_combined_sha256"]), 64)
        self.assertEqual(len(next(iter(m["input_files_sha256"].values()))), 64)

    def test_too_short_episode_is_reported_not_crashed(self):
        d = Path(tempfile.mkdtemp())
        write(d, "episode-001-short.txt", demo_episode(1, 15.0))
        recs, summ = run(d, Path(tempfile.mkdtemp()), log=lambda *_: None)
        self.assertEqual(recs, [])
        self.assertEqual(summ["episodes_without_promos"][0]["episode"], 1)


if __name__ == "__main__":
    unittest.main()
