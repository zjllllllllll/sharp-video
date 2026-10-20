import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaussian_correspondence import build_correspondence
from flow_worker import prepare_image


class GaussianCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.grid = {
            "grid_shape_lhw": [2, 4, 5], "image_size_wh": [10, 8],
            "grid_to_rgb_sample": torch.tensor([[2., 0, .5], [0, 2., .5], [0, 0, 1.]]),
        }
        self.zero = torch.zeros(2, 8, 10)
        self.depth = torch.full((8, 10), 10.)

    def build(self, forward=None, backward=None, target_depth=None, target_grid=None):
        return build_correspondence(
            self.zero if forward is None else forward,
            self.zero if backward is None else backward,
            self.depth, self.depth if target_depth is None else target_depth,
            self.grid, self.grid if target_grid is None else target_grid,
        )

    def test_zero_flow_is_identity(self):
        result = self.build()
        torch.testing.assert_close(result["candidate_target_index"], torch.arange(20, dtype=torch.int32).reshape(4, 5))
        self.assertTrue(result["in_bounds"].all())
        self.assertEqual(result["fb_error_px"].max(), 0)
        self.assertEqual(result["depth_relative_error"].max(), 0)

    def test_translation_moves_one_grid_cell_and_marks_boundary(self):
        forward = self.zero.clone()
        forward[0] = 2
        result = self.build(forward, -forward)
        expected = torch.arange(20, dtype=torch.int32).reshape(4, 5) + 1
        expected[:, -1] = -1
        torch.testing.assert_close(result["candidate_target_index"], expected)
        torch.testing.assert_close(result["target_grid_uv"][..., 0], torch.arange(1., 6.).expand(4, 5))
        self.assertTrue(torch.isnan(result["fb_error_px"][:, -1]).all())
        self.assertEqual(result["fb_error_px"][:, :-1].max(), 0)

    def test_backward_is_sampled_at_forward_destination(self):
        forward, backward = self.zero.clone(), self.zero.clone()
        forward[0] = torch.arange(10.) * .25
        backward[0] = torch.arange(10.) * -.2
        result = self.build(forward, backward)
        errors = result["fb_error_px"][result["in_bounds"]]
        self.assertLess(errors.max(), 2e-6)

    def test_errors_do_not_filter_candidates(self):
        backward = self.zero.clone()
        backward[0] = 100
        result = self.build(backward=backward, target_depth=self.depth * 3)
        self.assertTrue(result["in_bounds"].all())
        self.assertTrue((result["candidate_target_index"] >= 0).all())
        torch.testing.assert_close(result["fb_error_px"], torch.full((4, 5), 100.))
        torch.testing.assert_close(result["depth_relative_error"], torch.full((4, 5), 2 / 3))

    def test_target_grid_uses_its_own_inverse_transform(self):
        target = dict(self.grid)
        target["grid_to_rgb_sample"] = torch.eye(3)
        target["grid_shape_lhw"] = [2, 8, 10]
        result = self.build(target_grid=target)
        torch.testing.assert_close(result["target_grid_uv"][0, 0], torch.tensor([.5, .5]))
        torch.testing.assert_close(result["target_grid_uv"][3, 4], torch.tensor([8.5, 6.5]))

    def test_depth_is_sampled_at_corresponding_pixels(self):
        forward = self.zero.clone()
        forward[0] = 2
        target_depth = torch.arange(10.).expand(8, 10) + 10
        result = self.build(forward, -forward, target_depth)
        # Cell (0,0) samples p=(.5,.5), q=(2.5,.5): depths 10 and 12.5.
        self.assertAlmostEqual(result["depth_relative_error"][0, 0].item(), .2, places=6)

    def test_missing_depth_is_reported_without_removing_candidate(self):
        result = self.build(target_depth=torch.zeros(8, 10))
        self.assertTrue(torch.isnan(result["depth_relative_error"]).all())
        self.assertTrue((result["candidate_target_index"] >= 0).all())

    def test_nan_flow_does_not_create_a_candidate(self):
        forward = torch.full_like(self.zero, float("nan"))
        result = self.build(forward)
        self.assertTrue((result["candidate_target_index"] == -1).all())
        self.assertTrue(torch.isnan(result["fb_error_px"]).all())

    def test_preprocessing_only_pads_and_normalizes(self):
        import numpy as np
        image = np.zeros((9, 11, 3), dtype=np.uint8)
        image[-1, -1] = 255
        actual = prepare_image(image, "cpu")
        self.assertEqual(tuple(actual.shape), (1, 3, 16, 16))
        self.assertEqual(actual[0, 0, 0, 0].item(), -1)
        self.assertEqual(actual[0, 0, 8, 10].item(), 1)
        self.assertEqual(actual[0, 0, 15, 15].item(), 1)


if __name__ == "__main__":
    unittest.main()
