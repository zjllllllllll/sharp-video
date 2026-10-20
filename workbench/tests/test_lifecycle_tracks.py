import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaussian_animation import COLUMNS
from gaussian_lifecycle import LifecycleTrajectoryState, read_lifecycle_animation
from lifecycle_worker import run


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.grid = {"grid_shape_lhw": [2, 2, 4], "image_size_wh": [8, 4],
                     "grid_to_rgb_sample": torch.tensor([[2., 0, .5], [0, 2., .5], [0, 0, 1.]]),
                     "intrinsics": torch.eye(3), "coordinate_system": "camera_opencv_x_right_y_down_z_forward"}
        self.flow = torch.zeros(2, 4, 8)

    def test_identity_preserves_ids_and_lifetimes(self):
        state = LifecycleTrajectoryState(self.grid)
        self.assertEqual(state.advance(self.flow), 0)
        np.testing.assert_array_equal(state.array(), np.arange(8))
        np.testing.assert_array_equal(state.ids.flatten(), np.arange(8))
        np.testing.assert_array_equal(state.lifetimes(), np.column_stack((np.arange(8), np.zeros(8), np.full(8, 2))))

    def test_duplicate_targets_end_losers_and_fill_every_missing_cell(self):
        state = LifecycleTrajectoryState(self.grid)
        self.flow[0] = 2.5 - torch.arange(8.)
        self.assertEqual(state.advance(self.flow), 6)
        np.testing.assert_array_equal(state.array(), [1, 0, 2, 3, 5, 4, 6, 7])
        np.testing.assert_array_equal(state.ids.flatten(), [0, 8, 9, 10, 4, 11, 12, 13])
        np.testing.assert_array_equal(np.sort(state.array()), np.arange(8))
        lifetimes = state.lifetimes()
        np.testing.assert_array_equal(lifetimes[[1, 2, 3, 5, 6, 7], 2], 1)
        np.testing.assert_array_equal(lifetimes[8:, 1:], np.tile([1, 2], (6, 1)))

    def test_smallest_id_wins_even_when_newborn_has_earlier_slot(self):
        state = LifecycleTrajectoryState(self.grid)
        self.flow[0] = -2
        state.advance(self.flow)  # slots 0 and 4 now contain new, larger IDs
        self.flow[0] = 2.5 - torch.arange(8.)
        state.advance(self.flow)
        self.assertEqual(int(state.ids[0, 1]), 1)
        self.assertEqual(int(state.ids[1, 1]), 5)
        self.assertGreaterEqual(int(state.ids[0, 0]), 10)

    def test_subpixel_coordinates_are_retained_for_survivors(self):
        state = LifecycleTrajectoryState(self.grid)
        self.flow[0] = .6
        state.advance(self.flow)
        state.advance(self.flow)
        self.assertEqual(int(state.ids[0, 0]), 0)
        self.assertEqual(int(state.indices[0, 0]), 1)
        self.assertAlmostEqual(float(state.uv[0, 0, 0]), 1.7, places=6)

    def test_outside_or_nonfinite_ends_old_ids_and_births_new_ones(self):
        for value in [20., float("nan"), float("inf")]:
            state = LifecycleTrajectoryState(self.grid)
            self.assertEqual(state.advance(torch.full_like(self.flow, value)), 8)
            np.testing.assert_array_equal(state.array(), np.arange(8))
            np.testing.assert_array_equal(state.ids.flatten(), np.arange(8, 16))
            self.assertEqual(state.advance(self.flow), 0)
            np.testing.assert_array_equal(state.lifetimes()[:8, 2], 1)

    def test_single_file_has_all_original_records_once_and_correct_lifetimes(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as isolated:
            root = Path(directory)
            gaussians, flow = root / "gaussians", root / "flow"
            gaussians.mkdir()
            flow.mkdir()
            torch.save(self.grid, gaussians / "grid.pt")
            records, originals = [], []
            for frame in range(3):
                values = np.arange(16 * 14, dtype="<f4").reshape(16, 14) + frame * 1000
                originals.append(values)
                filename = f"frame_{frame:05}.ply"
                header = "ply\nformat binary_little_endian 1.0\nelement vertex 16\n"
                header += "".join(f"property float {name}\n" for name in COLUMNS) + "end_header\n"
                (gaussians / filename).write_bytes(header.encode() + values.tobytes())
                records.append({"index": frame, "full": filename, "tensor": "grid.pt"})
            (gaussians / "gaussian.json").write_text(json.dumps({"frames": records, "fps": 25}))
            self.flow[0] = 2.5 - torch.arange(8.)
            np.save(flow / "collapse.npy", self.flow.permute(1, 2, 0).numpy())
            np.save(flow / "zero.npy", np.zeros((4, 8, 2), np.float32))
            (flow / "flow.json").write_text(json.dumps({"pairs": [
                {"sourceFrame": 0, "targetFrame": 1, "forward": "collapse.npy"},
                {"sourceFrame": 1, "targetFrame": 2, "forward": "zero.npy"}]}))
            with patch("torch.cuda.is_available", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                run(SimpleNamespace(run_dir=directory, output_dir=None))
            output = root / "tracks-lifecycle"
            self.assertEqual({p.name for p in output.iterdir()}, {"animation.4dgs", "tracks.json"})
            state = json.loads((output / "tracks.json").read_text())
            self.assertEqual(state["status"], "complete")
            path = Path(isolated, "only.4dgs")
            shutil.copyfile(output / "animation.4dgs", path)
            header, samples, lifetimes = read_lifecycle_animation(path)
            try:
                self.assertEqual(header["validCounts"], [16, 16, 16])
                self.assertEqual(header["birthCounts"], [16, 12, 0])
                self.assertEqual(header["endedCounts"], [0, 12, 0])
                self.assertEqual(header["totalTrackCount"], 28)
                self.assertEqual(header["timesSeconds"], [0, .04, .08])
                for frame in range(3):
                    for layer in range(2):
                        actual = samples[frame]["parameters"][layer * 8:(layer + 1) * 8]
                        # All 14 fields of each original Gaussian occur exactly once.
                        self.assertEqual(sorted(row.tobytes() for row in actual),
                                         sorted(row.tobytes() for row in originals[frame][layer * 8:(layer + 1) * 8]))
                for identity, start, end in lifetimes:
                    occurrences = np.where(samples["ids"] == identity)[0]
                    np.testing.assert_array_equal(occurrences, np.arange(start, end))
                self.assertEqual(path.stat().st_size, state["file"]["dataOffset"] + header["lifetimes"]["offset"] + lifetimes.nbytes)
            finally:
                samples._mmap.close()
                lifetimes._mmap.close()


if __name__ == "__main__":
    unittest.main()
