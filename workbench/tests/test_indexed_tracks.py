import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaussian_tracks import IndexedTrackWriter, IndexedTrajectoryState, load_tracks
from track_worker import run


class IndexedTrackTests(unittest.TestCase):
    def setUp(self):
        # Deliberately contains no depth or Gaussian attributes.
        self.grid = {
            "grid_shape_lhw": [2, 4, 5], "image_size_wh": [10, 8],
            "grid_to_rgb_sample": torch.tensor([[2., 0, .5], [0, 2., .5], [0, 0, 1.]]),
            "intrinsics": torch.eye(3),
            "coordinate_system": "camera_opencv_x_right_y_down_z_forward",
        }
        self.flow = torch.zeros(2, 8, 10)

    def test_identity_and_zero_flow_without_depth(self):
        track = IndexedTrajectoryState(self.grid)
        np.testing.assert_array_equal(track.array(), np.arange(20))
        track.advance(self.flow)
        np.testing.assert_array_equal(track.array(), np.arange(20))

    def test_translation_and_subpixel_carry(self):
        track = IndexedTrajectoryState(self.grid)
        self.flow[0] = .6
        track.advance(self.flow)
        np.testing.assert_array_equal(track.array().reshape(4, 5)[:, :4], np.arange(20).reshape(4, 5)[:, :4])
        track.advance(self.flow)
        np.testing.assert_array_equal(track.array().reshape(4, 5)[:, :4], np.arange(20).reshape(4, 5)[:, :4] + 1)
        torch.testing.assert_close(track.uv[:, :4, 0], torch.arange(4.).expand(4, 4) * 2 + 1.7)

    def test_round_uses_ties_to_even(self):
        track = IndexedTrajectoryState(self.grid)
        self.flow[0] = 1
        track.advance(self.flow)
        np.testing.assert_array_equal(track.array()[:5], [0, 2, 2, 4, -1])

    def test_grid_boundary_can_stop_inside_rgb(self):
        self.grid["grid_to_rgb_sample"][:2, 2] = 0
        track = IndexedTrajectoryState(self.grid)
        self.flow[0] = 1
        track.advance(self.flow)
        # x=9 is still in the RGB image, round(4.5)=4 is still in the grid.
        self.assertEqual(track.array()[4], 4)
        self.flow[0] = -.1
        self.flow[1] = .1
        # With a shifted grid, an RGB-valid destination can have no grid cell.
        self.grid["grid_to_rgb_sample"][0, 2] = 2
        track = IndexedTrajectoryState(self.grid)
        self.flow[0] = -1.1
        track.advance(self.flow)
        self.assertEqual(track.array()[0], -1)  # RGB .9 -> grid -.55 -> -1.

    def test_out_of_image_never_resumes(self):
        track = IndexedTrajectoryState(self.grid)
        self.flow[0] = 20
        track.advance(self.flow)
        track.advance(-self.flow)
        np.testing.assert_array_equal(track.array(), np.full(20, -1))
        self.assertTrue(torch.isnan(track.uv).all())

    def test_nonfinite_flow_never_resumes(self):
        for value in [float("nan"), float("inf")]:
            track = IndexedTrajectoryState(self.grid)
            track.advance(torch.full_like(self.flow, value))
            track.advance(self.flow)
            np.testing.assert_array_equal(track.array(), np.full(20, -1))

    def test_many_to_one_keeps_all_slots(self):
        track = IndexedTrajectoryState(self.grid)
        self.flow[0] = 4.5 - torch.arange(10.)
        track.advance(self.flow)
        expected = np.repeat(np.arange(4) * 5 + 2, 5)
        np.testing.assert_array_equal(track.array(), expected)

    def test_save_load_and_range_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = IndexedTrackWriter(directory, 2, 20, [0, .04])
            writer.write_frame(0, np.arange(20))
            writer.write_frame(1, np.full(20, -1))
            writer.close()
            Path(directory, "tracks.json").write_text(json.dumps({"arrays": writer.descriptors}), encoding="utf-8")
            _, arrays = load_tracks(directory)
            try:
                self.assertEqual(set(arrays), {"sourceCellIndices", "times"})
                self.assertEqual(arrays["sourceCellIndices"].dtype.str, "<i4")
                np.testing.assert_array_equal(arrays["times"], [0, .04])
                spec = writer.descriptors["sourceCellIndices"]
                with open(Path(directory, spec["file"]), "rb") as stream:
                    stream.seek(spec["dataOffset"] + spec["frameBytes"])
                    data = np.frombuffer(stream.read(spec["frameBytes"]), dtype="<i4")
                np.testing.assert_array_equal(data, arrays["sourceCellIndices"][1])
            finally:
                for array in arrays.values():
                    array._mmap.close()

    def test_worker_completes_with_no_depth_or_gaussian_tensor_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "gaussians").mkdir()
            (root / "flow").mkdir()
            torch.save(self.grid, root / "gaussians" / "grid.pt")
            frames = [{"index": i, "full": f"frame_{i}.ply", "tensor": "grid.pt"} for i in range(2)]
            for frame in frames:
                fields = "x y z f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3".split()
                header = "ply\nformat binary_little_endian 1.0\nelement vertex 40\n"
                header += "".join(f"property float {name}\n" for name in fields) + "end_header\n"
                (root / "gaussians" / frame["full"]).write_bytes(header.encode() + np.zeros((40, 14), dtype="<f4").tobytes())
            (root / "gaussians" / "gaussian.json").write_text(json.dumps({"frames": frames, "fps": 25}), encoding="utf-8")
            self.flow[0] = 2
            np.save(root / "flow" / "forward.npy", self.flow.permute(1, 2, 0).numpy())
            (root / "flow" / "flow.json").write_text(json.dumps({"pairs": [
                {"sourceFrame": 0, "targetFrame": 1, "forward": "forward.npy"}
            ]}), encoding="utf-8")
            with patch("torch.cuda.is_available", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                run(SimpleNamespace(run_dir=directory, output_dir=None))
            output = root / "tracks-indexed"
            manifest, arrays = load_tracks(output)
            try:
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(manifest["format"], "sharp.gaussian_tracks.v3")
                self.assertEqual(manifest["validCounts"], [40, 32])
                np.testing.assert_array_equal(arrays["sourceCellIndices"][0], np.arange(20))
                np.testing.assert_array_equal(arrays["sourceCellIndices"][1], [1, 2, 3, 4, -1, 6, 7, 8, 9, -1, 11, 12, 13, 14, -1, 16, 17, 18, 19, -1])
                self.assertEqual({p.name for p in output.iterdir()}, {"tracks.json", "source_cell_indices.npy", "times.npy"})
            finally:
                for array in arrays.values():
                    array._mmap.close()

    def test_missing_input_reports_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                run(SimpleNamespace(run_dir=directory, output_dir=None))
            state = json.loads(Path(directory, "tracks-indexed", "tracks.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "failed")
            self.assertIn("flow.json", state["error"])


if __name__ == "__main__":
    unittest.main()
