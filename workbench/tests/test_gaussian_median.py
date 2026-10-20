import sys
from pathlib import Path
import unittest

import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaussian_median import median_frame, rotation_medoid


class MedianTests(unittest.TestCase):
    def test_matches_scipy_per_lifetime_all_scalar_fields_and_layers(self):
        rng = np.random.default_rng(42)
        parameters = rng.normal(size=(13, 4, 14)).astype('float32')
        parameters[..., 10:] = [1, 0, 0, 0]
        parameters[4, 0, 6] = np.inf  # saturated opacity has a well-defined median
        ids = np.zeros((13, 2), dtype='uint32')
        ids[:, 1] = 1
        ids[2:, 0] = 2
        ids[3:, 0] = 3
        lives = np.array([[0,0,2], [1,0,13], [2,2,3], [3,3,13]], dtype='uint32')
        expected = parameters.copy()
        for slot, start, end in [(0,0,2),(0,2,3),(0,3,13),(1,0,13)]:
            if end - start < 5:
                continue
            for layer in range(2):
                row = slot + 2*layer
                expected[start:end,row,:10] = median_filter(parameters[start:end,row,:10], size=(5,1), mode='reflect')
        actual = np.stack([median_frame(t, ids[t], lives, lambda i: parameters[i], 2) for t in range(13)])
        np.testing.assert_array_equal(actual, expected)

    def test_short_tracks_preserve_every_attribute_and_five_frames_still_filter(self):
        rng = np.random.default_rng(7)
        for length in range(1, 6):
            with self.subTest(length=length):
                parameters = rng.normal(size=(length, 2, 14)).astype('float32')
                parameters[..., 10:] /= np.linalg.norm(parameters[..., 10:], axis=-1, keepdims=True)
                lives = np.array([[0, 0, length]], dtype='uint32')
                reads = []
                def read(t):
                    reads.append(t)
                    return parameters[t]
                actual = np.stack([median_frame(t, np.array([0]), lives, read, 2) for t in range(length)])
                if length < 5:
                    np.testing.assert_array_equal(actual.view('uint32'), parameters.view('uint32'))
                    self.assertEqual(reads, list(range(length)))
                else:
                    expected = median_filter(parameters[..., :10], size=(5, 1, 1), mode='reflect')
                    np.testing.assert_array_equal(actual[..., :10], expected)
                    self.assertFalse(np.array_equal(actual, parameters))

    def test_step_kept_and_two_frame_impulse_removed(self):
        x = np.ones((16, 1, 14), dtype='float32')
        x[..., 10:] = [1,0,0,0]
        x[3:5,0,:10] = 9
        x[9:,0,:10] = 4
        lives = np.array([[0,0,16]], dtype='uint32')
        actual = np.stack([median_frame(t, np.array([0]), lives, lambda i: x[i], 1) for t in range(16)])
        np.testing.assert_array_equal(actual[:9,0,:10], 1)
        np.testing.assert_array_equal(actual[9:,0,:10], 4)

    def test_rotation_angular_medoid_ignores_quaternion_sign(self):
        q = Rotation.from_euler('z', [0,1,120,2,3], degrees=True).as_quat(scalar_first=True)
        result = rotation_medoid(q[:,None])
        expected = q[3]
        np.testing.assert_allclose(abs(result @ expected), 1)
        q[[0,2,3]] *= -1
        flipped = rotation_medoid(q[:,None])
        np.testing.assert_allclose(abs(flipped @ expected), 1)
        np.testing.assert_allclose(np.linalg.norm(flipped, axis=-1), 1)


if __name__ == '__main__':
    unittest.main()
