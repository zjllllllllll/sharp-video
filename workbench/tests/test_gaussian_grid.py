import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

WORKBENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKBENCH))
sys.path.insert(0, str(WORKBENCH.parent / "ml-sharp" / "src"))

from gaussian_grid import (
    flat_index_to_grid, grid_to_gaussians, grid_uv_to_rgb_sample,
    load_grid_frame, make_grid_frame, project_gaussian_centers, save_grid_frame,
)
from sharp.models.composer import GaussianComposer
from sharp.models.initializer import create_initializer
from sharp.models.params import PredictorParams
from sharp.utils.gaussians import Gaussians3D, load_ply, save_ply


class GaussianGridTests(unittest.TestCase):
    def metadata(self):
        return dict(
            grid_shape_lhw=(2, 3, 4), image_size_wh=(19, 11),
            network_size_wh=(8, 6), focal_length_px=15.0,
            frame_index=7, fps=25.0, source={"kind": "prediction"},
        )

    def gaussians(self):
        torch.manual_seed(42)
        xyz = torch.rand(1, 24, 3)
        xyz[..., 2] += 2
        return Gaussians3D(
            mean_vectors=xyz,
            singular_values=torch.rand(1, 24, 3) * 0.2 + 0.01,
            quaternions=F.normalize(torch.randn(1, 24, 4), dim=-1),
            colors=torch.rand(1, 24, 3),
            opacities=torch.rand(1, 24) * 0.8 + 0.1,
        )

    def test_layout_matches_actual_sharp_composer(self):
        params = PredictorParams()
        initializer = create_initializer(params.initializer)
        base = initializer(torch.rand(1, 3, 6, 8), torch.ones(1, 2, 6, 8))
        composer = GaussianComposer(
            delta_factor=params.delta_factor, min_scale=params.min_scale,
            max_scale=params.max_scale, color_activation_type=params.color_activation_type,
            opacity_activation_type=params.opacity_activation_type,
            color_space=params.color_space, base_scale_on_predicted_mean=params.base_scale_on_predicted_mean,
        )
        delta = torch.linspace(-0.5, 0.5, 14 * 2 * 3 * 4).reshape(1, 14, 2, 3, 4)
        structured = composer(delta, base.gaussian_base_values, flatten_output=False)
        flat = composer(delta, base.gaussian_base_values)
        frame = make_grid_frame(flat, **self.metadata())
        for index in range(24):
            layer, row, column = flat_index_to_grid(index, (2, 3, 4))
            for name in flat._fields:
                expected = (getattr(structured, name)[0, layer, row, column]
                            if name == "opacities" else
                            getattr(structured, name)[0, :, layer, row, column])
                torch.testing.assert_close(frame["gaussians"][name][layer, row, column], expected, rtol=0, atol=0)

    def test_sampling_coordinates_match_actual_resize_and_pool(self):
        # A coordinate ramp remains linear under bilinear resizing. Its pooled
        # values give the original-image sampling centers independently.
        xx, yy = torch.meshgrid(torch.arange(19), torch.arange(11), indexing="xy")
        ramp = torch.stack((xx, yy)).to(torch.float64)[None]
        actual = F.avg_pool2d(F.interpolate(ramp, size=(6, 8), mode="bilinear", align_corners=True), 2)
        u, v = torch.meshgrid(torch.arange(4), torch.arange(3), indexing="xy")
        frame = make_grid_frame(self.gaussians(), **self.metadata())
        mapped = grid_uv_to_rgb_sample(torch.stack((u, v), dim=-1), frame)
        torch.testing.assert_close(mapped, actual[0].permute(1, 2, 0), rtol=1e-14, atol=1e-14)

    def test_save_load_preserves_every_attribute_exactly(self):
        original = self.gaussians()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.pt"
            save_grid_frame(path, original, **self.metadata())
            frame = load_grid_frame(path)
            restored = grid_to_gaussians(frame)
        self.assertEqual(frame["time_seconds"], 7 / 25)
        self.assertEqual(frame["grid_shape_lhw"], [2, 3, 4])
        for name in original._fields:
            torch.testing.assert_close(getattr(restored, name), getattr(original, name), rtol=0, atol=0)

    def test_ply_recovery_keeps_order_and_decodes_attributes(self):
        original = self.gaussians()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.ply"
            save_ply(original, 15.0, (11, 19), path)
            recovered, _ = load_ply(path)
            frame = make_grid_frame(recovered, **self.metadata())
            restored = grid_to_gaussians(frame)
        for name in original._fields:
            torch.testing.assert_close(getattr(restored, name), getattr(original, name), rtol=2e-6, atol=2e-7)

    def test_projected_center_uses_position_not_cell_index(self):
        original = self.gaussians()
        original.mean_vectors[0, 0] = torch.tensor([1.0, -0.5, 2.0])
        frame = make_grid_frame(original, **self.metadata())
        projected = project_gaussian_centers(frame)
        torch.testing.assert_close(projected[0, 0, 0], torch.tensor([17.0, 1.75]))
        self.assertFalse(torch.allclose(projected[0, 0, 0].double(), grid_uv_to_rgb_sample([0, 0], frame)))


if __name__ == "__main__":
    unittest.main()
