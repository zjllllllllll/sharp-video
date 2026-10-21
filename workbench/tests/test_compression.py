import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compression_worker import pack_ply
from trajectory_codec.legacy_format import write_header, read_lifecycle_animation, make_lifecycle_header


class CompressionTests(unittest.TestCase):
    def test_ply_adapter_preserves_order_sh0_scale_and_saturated_opacity(self):
        names = 'x y z f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3'.split()
        v = np.zeros(2, dtype=[(name, '<f4') for name in reversed(names)])
        v['x'], v['y'], v['z'] = [3, -1], [2, 7], [1, 4]
        v['opacity'] = [-np.inf, np.inf]
        v['scale_0'], v['scale_1'], v['scale_2'] = np.log([.01, .02]), np.log([.02, .04]), np.log([.03, .06])
        v['rot_0'] = 1
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'frame.ply'
            PlyData([PlyElement.describe(v,'vertex')],text=False,byte_order='<').write(p)
            packed = pack_ply(p)
        assert packed.shape == (2,32)
        floats = packed.copy().view('<f4').reshape(2,8)
        np.testing.assert_array_equal(floats[:,:3],[[3,2,1],[-1,7,4]])
        np.testing.assert_allclose(floats[:,3:6],[[.01,.02,.03],[.02,.04,.06]],rtol=1e-6)
        np.testing.assert_array_equal(packed[:,24:27],np.full((2,3),128))
        np.testing.assert_array_equal(packed[:,27],[0,255])
        np.testing.assert_array_equal(packed[:,28:32],[[255,128,128,128],[255,128,128,128]])

    def test_separate_codec_legacy_reader_remains_self_contained(self):
        metadata={'trackCount':2,'cellCount':1,'layerCount':2,'frameCount':1,'fps':25,
                  'birthCounts':[2],'endedCounts':[0]}
        lifetimes=np.array([[0,0,1]],dtype='<u4')
        header=make_lifecycle_header(metadata,np.array([0.0]),lifetimes)
        parameters=np.arange(28,dtype='<f4').reshape(2,14)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'legacy.4dgs'
            with p.open('wb') as f:
                write_header(f,header)
                f.write(np.array([0],dtype='<u4').tobytes())
                f.write(parameters.tobytes())
                f.write(lifetimes.tobytes())
            h,frames,lives=read_lifecycle_animation(p)
            np.testing.assert_array_equal(frames['parameters'][0],parameters)
            np.testing.assert_array_equal(lives,lifetimes)
            frames._mmap.close(); lives._mmap.close()


if __name__ == '__main__': unittest.main()
