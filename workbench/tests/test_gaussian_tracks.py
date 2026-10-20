import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaussian_tracks import TrackWriter, TrajectoryState, load_tracks, reference_attributes


class GaussianTrackTests(unittest.TestCase):
    def setUp(self):
        self.grid = {
            "grid_shape_lhw": [2, 4, 5], "image_size_wh": [10, 8],
            "grid_to_rgb_sample": torch.tensor([[2., 0, .5], [0, 2., .5], [0, 0, 1.]]),
            "intrinsics": torch.tensor([[10., 0, 5.], [0, 10., 4.], [0, 0, 1.]]),
            "gaussians": {
                "mean_vectors": torch.arange(120.).reshape(2, 4, 5, 3),
                "singular_values": torch.ones(2, 4, 5, 3),
                "quaternions": torch.tensor([1., 0, 0, 0]).expand(2, 4, 5, 4).clone(),
                "colors": torch.ones(2, 4, 5, 3), "opacities": torch.ones(2, 4, 5),
            },
        }
        self.depth = torch.full((8, 10), 2.)
        self.flow = torch.zeros(2, 8, 10)

    def test_reference_geometry_and_zero_motion(self):
        track = TrajectoryState(self.grid, self.depth)
        track.advance(self.flow, self.depth)
        torch.testing.assert_close(track.positions, self.grid["gaussians"]["mean_vectors"])
        self.assertTrue(track.valid.all())

    def test_subpixel_carry_and_known_3d_displacement(self):
        track = TrajectoryState(self.grid, self.depth)
        self.flow[0] = .125
        for _ in range(4):
            track.advance(self.flow, self.depth)
        torch.testing.assert_close(track.uv[..., 0], torch.arange(1., 10., 2.).expand(4, 5))
        expected = self.grid["gaussians"]["mean_vectors"] + torch.tensor([.1, 0, 0])
        torch.testing.assert_close(track.positions, expected)

    def test_depth_change_preserves_initial_offset(self):
        track = TrajectoryState(self.grid, self.depth)
        original_uv = track.uv.clone()
        track.advance(self.flow, self.depth * 4)
        ray = torch.cat(((original_uv - torch.tensor([5., 4.])) / 10, torch.ones(4, 5, 1)), -1)
        torch.testing.assert_close(track.positions, self.grid["gaussians"]["mean_vectors"] + ray * 6)
        self.assertTrue(track.valid.all())  # Large depth changes are not filtered.

    def test_missing_depth_can_recover_without_reassigning_ids(self):
        track = TrajectoryState(self.grid, self.depth)
        self.flow[0] = .1
        track.advance(self.flow, self.depth * 0)
        self.assertFalse(track.valid.any())
        self.assertTrue(torch.isnan(track.positions).all())
        self.assertTrue(track.tracked.all())
        track.advance(self.flow, self.depth)
        self.assertTrue(track.valid.all())
        torch.testing.assert_close(track.positions, self.grid["gaussians"]["mean_vectors"] + torch.tensor([.04, 0, 0]))

    def test_outside_image_never_reenters_as_another_track(self):
        track = TrajectoryState(self.grid, self.depth)
        self.flow[0] = 20
        track.advance(self.flow, self.depth)
        track.advance(-self.flow, self.depth)
        self.assertFalse(track.valid.any())
        self.assertTrue(torch.isnan(track.uv).all())

    def test_missing_reference_or_flow_remains_undefined(self):
        track = TrajectoryState(self.grid, self.depth * 0)
        track.advance(self.flow, self.depth)
        self.assertFalse(track.valid.any())
        track = TrajectoryState(self.grid, self.depth)
        self.flow[:] = torch.nan
        track.advance(self.flow, self.depth)
        self.assertFalse(track.valid.any())

    def test_many_to_one_does_not_merge_ids(self):
        track = TrajectoryState(self.grid, self.depth)
        self.flow[0] = 5 - torch.arange(10.)
        track.advance(self.flow, self.depth)
        torch.testing.assert_close(track.uv[..., 0], torch.full((4, 5), 5.))
        self.assertEqual(tuple(track.positions.shape), (2, 4, 5, 3))
        self.assertEqual(track.arrays()["displacements"].shape, (20, 3))
        self.assertTrue(track.valid.all())

    def test_both_layers_keep_attributes_and_relative_positions(self):
        self.grid["gaussians"]["colors"][1] *= .3
        self.grid["gaussians"]["singular_values"][1] *= 2
        self.grid["gaussians"]["opacities"][1] *= .7
        self.grid["gaussians"]["quaternions"][1] = torch.tensor([0., 1., 0., 0.])
        track = TrajectoryState(self.grid, self.depth)
        self.flow[0] = .25
        track.advance(self.flow, self.depth * 3)
        initial = self.grid["gaussians"]["mean_vectors"]
        torch.testing.assert_close(track.positions[1] - track.positions[0], initial[1] - initial[0])
        reference = reference_attributes(self.grid)
        self.assertEqual(reference.shape, (40, 14))
        for name, start, stop in [("mean_vectors", 0, 3), ("singular_values", 3, 6),
                                  ("quaternions", 6, 10), ("colors", 10, 13), ("opacities", 13, 14)]:
            np.testing.assert_array_equal(reference[:, start:stop], self.grid["gaussians"][name].reshape(40, stop - start).numpy())

    def test_undefined_center_does_not_hide_other_layer(self):
        self.grid["gaussians"]["mean_vectors"][1, 0, 0] = torch.nan
        track = TrajectoryState(self.grid, self.depth)
        track.advance(self.flow, self.depth)
        self.assertTrue(track.motion_valid[0, 0])
        self.assertTrue(track.valid[0, 0, 0])
        self.assertFalse(track.valid[1, 0, 0])
        self.assertTrue(torch.isnan(track.positions[1, 0, 0]).all())

    def test_save_load_and_frame_byte_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = reference_attributes(self.grid)
            writer = TrackWriter(directory, 2, 40, reference, [0, .04], motion_count=20)
            track = TrajectoryState(self.grid, self.depth)
            writer.write_frame(0, track.arrays())
            self.flow[0] = 1
            track.advance(self.flow, self.depth)
            writer.write_frame(1, track.arrays())
            writer.flush()
            Path(directory, "tracks.json").write_text(json.dumps({"arrays": writer.descriptors}), encoding="utf-8")
            _, loaded = load_tracks(directory)
            np.testing.assert_array_equal(loaded["ids"], np.arange(40))
            np.testing.assert_array_equal(loaded["reference"], reference)
            np.testing.assert_array_equal(loaded["displacements"][1], track.arrays()["displacements"])
            # The renderer selects motion id = gaussian id % motion_count.
            restored = reference[:, :3] + loaded["displacements"][1][loaded["ids"] % 20]
            np.testing.assert_array_equal(restored, track.positions.reshape(40, 3).numpy())
            self.assertEqual(loaded["uv"].shape, (2, 20, 2))
            self.assertEqual(loaded["valid"].shape, (2, 20))
            spec = writer.descriptors["displacements"]
            with open(Path(directory, spec["file"]), "rb") as stream:
                stream.seek(spec["dataOffset"] + spec["frameBytes"])
                values = np.frombuffer(stream.read(spec["frameBytes"]), dtype="<f4").reshape(20, 3)
            np.testing.assert_array_equal(values, loaded["displacements"][1])
            for array in [*loaded.values(), *writer.arrays.values()]:
                array._mmap.close()


if __name__ == "__main__":
    unittest.main()
