import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from animation_worker import run
from gaussian_animation import COLUMNS, FORMAT, gather_parameters, read_animation


class GaussianAnimationTests(unittest.TestCase):
    def fixture(self, root):
        gaussians, indexed = root / "gaussians", root / "tracks-indexed"
        gaussians.mkdir()
        indexed.mkdir()
        frames = []
        originals = []
        for frame in range(3):
            values = np.arange(6 * 14, dtype="<f4").reshape(6, 14) / 20
            values += frame * np.arange(1, 7, dtype="<f4")[:, None]
            originals.append(values)
            name = f"frame_{frame:05}.ply"
            header = "ply\nformat binary_little_endian 1.0\nelement vertex 6\n"
            header += "".join(f"property float {name}\n" for name in COLUMNS)
            header += "element version 3\nproperty uchar version\nend_header\n"
            (gaussians / name).write_bytes(header.encode() + values.tobytes() + bytes([1, 0, 0]))
            frames.append({"index": frame, "full": name})
        indices = np.array([[0, 1, 2], [2, 2, -1], [1, 0, -1]], dtype="<i4")
        np.save(indexed / "indices.npy", indices)
        np.save(indexed / "times.npy", np.array([0, .04, .08], dtype="<f8"))
        tracks = {"status": "complete", "frameCount": 3, "trackCount": 6, "cellCount": 3, "layerCount": 2,
                  "layers": [0, 1], "fps": 25, "frameIndices": [0, 1, 2], "validCounts": [6, 4, 4],
                  "sourceFrames": frames, "arrays": {"sourceCellIndices": {"file": "indices.npy"}, "times": {"file": "times.npy"}}}
        (indexed / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")
        return originals, indices

    def test_export_independent_file_preserves_all_parameters(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as isolated:
            root = Path(directory)
            originals, indices = self.fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                run(SimpleNamespace(run_dir=directory, output_dir=None))
            state = json.loads((root / "tracks-animation" / "tracks.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["format"], FORMAT)
            # Only the file is copied. No manifest, PLY, indices, depth or flow.
            standalone = Path(isolated) / "only-file.4dgs"
            shutil.copyfile(root / "tracks-animation" / "animation.4dgs", standalone)
            header, samples = read_animation(standalone)
            try:
                self.assertEqual(header["timesSeconds"], [0, .04, .08])
                self.assertEqual(header["reference"]["shape"], [6, 14])
                self.assertEqual(header["samples"]["shape"], [2, 6, 14])
                self.assertEqual(samples[0].tobytes(), originals[0].tobytes())
                for frame in range(3):
                    for layer in range(2):
                        for cell in range(3):
                            row = samples[frame, layer * 3 + cell]
                            target = indices[frame, cell]
                            if target >= 0:
                                self.assertEqual(row.tobytes(), originals[frame][layer * 3 + target].tobytes())
                            else:
                                self.assertEqual(row[6], -np.inf)
                                self.assertTrue(np.isfinite(np.delete(row, 6)).all())
                                self.assertEqual(row[10], 1)
                self.assertNotEqual(samples[1, 0, 0] - samples[0, 0, 0], samples[1, 3, 0] - samples[0, 3, 0])
                self.assertNotIn("sourceFrames", header)
                self.assertNotIn("arrays", header)
            finally:
                samples._mmap.close()

    def test_missing_source_reports_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                run(SimpleNamespace(run_dir=directory, output_dir=None))
            state = json.loads(Path(directory, "tracks-animation", "tracks.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "failed")
            self.assertFalse(Path(directory, "tracks-animation", "animation.4dgs").exists())


if __name__ == "__main__":
    unittest.main()
