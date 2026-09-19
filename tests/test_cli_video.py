import unittest

from go1_hier_sim.cli import _video_frame_count


class CliVideoTests(unittest.TestCase):
    def test_video_duration_converts_to_control_frames(self):
        self.assertEqual(_video_frame_count(60.0, 0.02), 3000)

    def test_video_duration_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            _video_frame_count(0.0, 0.02)
        with self.assertRaisesRegex(ValueError, "positive"):
            _video_frame_count(1.0, 0.0)


if __name__ == "__main__":
    unittest.main()
