import unittest

from _common import episode_from, filler, idx_for  # noqa: E402
from promofinder import scoring as S
from promofinder.validate import validate_segments


def codes(ix, segs):
    return {v.code for v in validate_segments(ix, segs)}


# The six real failures reported against v1, rebuilt as SYNTHETIC episodes with the same numbers.
# (episode label, segment start, segment end, dialogue interval crossed, which boundary is bad)
SIX = [("rank16", 32.5, 69.5, (68.0, 70.5), "end"), ("rank18", 48.8, 98.2, (96.5, 99.2), "end"),
       ("rank19", 63.0, 85.5, (62.4, 63.5), "start"), ("rank84", 0.0, 29.5, (29.2, 30.8), "end"),
       ("rank90", 76.5, 106.0, (104.2, 106.2), "end"), ("rank96", 48.8, 102.5, (98.3, 103.2), "end")]


def six_case_episode(a, b, dlg, side):
    ds, de = dlg
    ev = [("MARA", ds, de, "Invented line that spans the cut.")]
    ev.append(("ACTION", a, min(a + 2.0, b - 1), "An invented action begins."))         # makes `a` a source start
    ev.append(("ACTION", max(b - 2.0, a + 1), b, "An invented action ends."))            # makes `b` a source end
    skip = [(ds - 0.1, de + 0.1), (a - 0.1, a + 2.2), (b - 2.2, b + 0.1)]
    ev += filler(0.0, b + 12.0, skip=skip)
    return episode_from(ev, scene_end=b + 14.0)


class SixRealFailures(unittest.TestCase):
    def test_each_original_cut_is_rejected(self):
        for name, a, b, dlg, side in SIX:
            with self.subTest(name):
                ep = six_case_episode(a, b, dlg, side)
                ix = idx_for(ep)
                got = validate_segments(ix, [(ep.num, a, b)])
                self.assertIn("boundary_inside_dialogue", {v.code for v in got}, [str(v) for v in got])

    def test_generator_never_emits_a_cut_inside_dialogue(self):
        for name, a, b, dlg, side in SIX:
            with self.subTest(name):
                ep = six_case_episode(a, b, dlg, side)
                singles = S.generate_single(ep)
                self.assertTrue(singles, "fixture should still allow legal windows")
                pool = singles + S.generate_stitched(ep, singles)
                legal = [c for c in pool if S.is_valid(ep, c)]
                for c in legal:
                    for s in c.segments:
                        self.assertFalse(dlg[0] < s.start < dlg[1] or dlg[0] < s.end < dlg[1], (s.start, s.end))
                chosen, _ = S.find_candidates(ep, per_ep=5)
                for c in chosen:
                    self.assertEqual(validate_segments(idx_for(ep), [(ep.num, s.start, s.end) for s in c.segments]), [])

    def test_moving_the_cut_to_the_dialogue_edge_is_legal(self):
        ep = six_case_episode(32.5, 69.5, (68.0, 70.5), "end")
        self.assertEqual([str(v) for v in validate_segments(idx_for(ep), [(ep.num, 32.5, 70.5)])], [])


class DurationAndStructure(unittest.TestCase):
    def setUp(self):
        ev = [("ACTION", 0.0, 1.0, "Start."), ("A", 1.0, 3.0, "Line."), ("ACTION", 20.0, 21.0, "x"), ("A", 21.0, 22.0, "y"),
              ("A", 30.0, 31.0, "z"), ("ACTION", 60.0, 61.0, "w"), ("A", 61.0, 62.0, "w2"), ("A", 80.0, 81.0, "v"), ("A", 90.0, 91.0, "u"),
              ("ACTION", 95.0, 96.0, "t")]
        self.ep = episode_from(ev, scene_end=100.0)
        self.ix = idx_for(self.ep)
        self.n = self.ep.num

    def test_duration_limits_are_inclusive_and_exact(self):
        self.assertEqual(codes(self.ix, [(self.n, 1.0, 21.0)]), set())           # exactly 20.00 s: accepted
        self.assertIn("duration_out_of_range", codes(self.ix, [(self.n, 3.0, 22.0)]))   # 19.00 s: rejected
        self.assertEqual(codes(self.ix, [(self.n, 1.0, 61.0)]), set())           # exactly 60.00 s: accepted
        self.assertIn("duration_out_of_range", codes(self.ix, [(self.n, 1.0, 62.0)]))    # 61.00 s: rejected

    def test_non_source_boundary_and_bounds(self):
        self.assertIn("end_not_source_boundary", codes(self.ix, [(self.n, 1.0, 40.37)]))
        self.assertIn("out_of_episode_bounds", codes(self.ix, [(self.n, 80.0, 120.0)]))
        self.assertIn("non_positive_length", codes(self.ix, [(self.n, 30.0, 30.0)]))
        self.assertIn("not_finite", codes(self.ix, [(self.n, float("nan"), 30.0)]))
        self.assertIn("unknown_episode", codes(self.ix, [(99, 1.0, 30.0)]))
        self.assertIn("empty", codes(self.ix, []))

    def test_stitched_segments(self):
        ok = [(self.n, 1.0, 22.0), (self.n, 80.0, 81.0)]                  # 21 s + 1 s = 22 s, legal cut points
        self.assertEqual([str(v) for v in validate_segments(self.ix, ok)], [])
        rev = [(self.n, 80.0, 81.0), (self.n, 1.0, 22.0)]
        self.assertIn("segments_not_chronological", codes(self.ix, rev))
        overlap = [(self.n, 1.0, 22.0), (self.n, 21.0, 31.0)]
        self.assertIn("segments_not_chronological", codes(self.ix, overlap))
        short = [(self.n, 1.0, 3.0), (self.n, 21.0, 22.0)]
        self.assertIn("duration_out_of_range", codes(self.ix, short))                 # total of both is what counts

    def test_stitched_total_duration_counts_all_segments(self):
        segs = [(self.n, 1.0, 22.0), (self.n, 80.0, 91.0)]     # 21 + 11 = 32 s
        self.assertNotIn("duration_out_of_range", codes(self.ix, segs))


class OverlappingTracks(unittest.TestCase):
    def test_overlapping_dialogue_protects_both_lines(self):
        ev = [("A", 10.0, 14.0, "First long line."), ("B", 12.0, 16.0, "Interrupting line."), ("ACTION", 0.0, 1.0, "s"),
              ("ACTION", 40.0, 41.0, "e")]
        ep = episode_from(ev, scene_end=45.0)
        ix = idx_for(ep)
        # 14.0 is the END of line A but lies inside line B (12-16): illegal. 16.0 is legal.
        self.assertIn("boundary_inside_dialogue", codes(ix, [(ep.num, 0.0, 14.0)]))
        self.assertNotIn(14.0, ix[ep.num].cut_ends())
        self.assertIn(16.0, ix[ep.num].cut_ends())

    def test_camera_and_action_tracks_may_be_crossed(self):
        ev = [("ACTION", 0.0, 30.0, "A long action."), ("CAMERA", 5.0, 25.0, "Long shot."), ("A", 2.0, 4.0, "Hi."),
              ("A", 26.0, 28.0, "Bye."), ("ACTION", 24.0, 25.0, "Beat.")]
        ep = episode_from(ev, scene_end=30.0)
        # cut 2.0-25.0 crosses the long action and camera shot: allowed by design (only dialogue is protected)
        self.assertEqual(validate_segments(idx_for(ep), [(ep.num, 2.0, 25.0)]), [])


if __name__ == "__main__":
    unittest.main()
