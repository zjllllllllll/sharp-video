import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trajectory_codec.legacy_format import write_header
from trajectory_codec.legacy_format import make_lifecycle_header, read_lifecycle_animation
from trajectory_codec.data import prepare, epoch_batches
from trajectory_codec.model import (AttributeDecoder, FORMAT, default_config, group_losses,
                                    local_times, physical_attributes, STANDARD_COLUMNS)


def fixture(path):
    lifetimes = np.array([[0, 0, 3], [1, 0, 1], [2, 1, 2], [3, 2, 3]], dtype='<u4')
    metadata = dict(frameCount=3, cellCount=2, layerCount=2, trackCount=4, fps=25.,
                    frameIndices=[0, 1, 2], imageSizeWH=[8, 4], intrinsics=[[4,0,4],[0,4,2],[0,0,1]],
                    birthCounts=[4,2,2], endedCounts=[0,2,2])
    header = make_lifecycle_header(metadata, [0., .04, .12], lifetimes)
    ids = [[0,1], [0,2], [0,3]]
    with path.open('wb') as stream:
        write_header(stream, header)
        for frame in range(3):
            np.array(ids[frame], dtype='<u4').tofile(stream)
            p = np.random.default_rng(frame).normal(size=(4,14)).astype('<f4')
            p[:, 2] += 3
            p[:, 10:14] /= np.linalg.norm(p[:,10:14], axis=1, keepdims=True)
            p[0,6], p[1,6] = -np.inf, np.inf
            p.tofile(stream)
        lifetimes.tofile(stream)
    return header


class CodecTests(unittest.TestCase):
    def test_upstream_batched_output_and_gradients(self):
        torch.manual_seed(12)
        model = AttributeDecoder(default_config()).double()
        upstream = copy.deepcopy(model)
        z = torch.randn(3,64,dtype=torch.float64,requires_grad=True)
        t = torch.randn(3,1,dtype=torch.float64,requires_grad=True)
        other_z, other_t = z.detach().clone().requires_grad_(), t.detach().clone().requires_grad_()
        actual = model(z,t)
        expected = torch.cat([upstream.net(other_t[i:i+1], upstream.modulator(other_z[i])) for i in range(3)])
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(z.grad,other_z.grad,rtol=1e-9,atol=1e-10)
        torch.testing.assert_close(t.grad,other_t.grad,rtol=1e-9,atol=1e-10)
        for a,b in zip(model.parameters(),upstream.parameters()):
            torch.testing.assert_close(a.grad,b.grad,rtol=1e-9,atol=1e-10)
        self.assertEqual(sum(p.numel() for p in model.parameters()),612622)

    def test_loss_handles_saturation_and_quaternion_sign(self):
        pred = torch.randn(3,14,requires_grad=True)
        target = torch.randn(3,14)
        target[0,6],target[1,6] = -torch.inf,torch.inf
        mean,std = torch.zeros(9),torch.ones(9)
        first = group_losses(pred,target,mean,std)
        target[:,10:14] *= -1
        torch.testing.assert_close(first,group_losses(pred,target,mean,std))
        first.sum().backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        q = physical_attributes(pred.detach(),mean,std)[:,10:14]
        torch.testing.assert_close(q.norm(dim=-1),torch.ones(3))

    def test_lifecycle_sampling_resume_and_independent_decode(self):
        cache = Path(__file__).resolve().parents[2] / '.cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache, prefix='codec-test-') as directory:
            root = Path(directory)
            assert root.resolve().is_relative_to(cache.resolve())
            source = root/'source.4dgs'
            header = fixture(source)
            prepared = prepare(source)
            life = prepared['lifecycles'].numpy()
            np.testing.assert_array_equal(life,[[0,3,0],[0,1,1],[1,2,1],[2,3,1]])
            np.testing.assert_allclose(local_times(np.array([0,2]),1,life,np.array([0.,.04,.12])),[[-1/3],[0]],atol=1e-7)
            config = default_config()
            config['batch_size']=2
            _,frames,lifetimes=read_lifecycle_animation(source)
            batches=list(epoch_batches(header,frames,life,config,0))
            resumed=list(epoch_batches(header,frames,life,config,0,skip=3))
            self.assertEqual(len(batches),6)
            for a,b in zip(batches[3:],resumed):
                for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
            visits=[]
            for _,frame,ids,_,_ in batches: visits.extend((frame,int(i)) for i in ids)
            self.assertEqual(len(set(visits)),12)
            self.assertEqual({i for _,i in visits},set(range(8)))
            frames._mmap.close();lifetimes._mmap.close()
            network=AttributeDecoder(config)
            package=dict(format=FORMAT,config=config,network=network.state_dict(),codes=torch.randn(8,64),
                         header=header,mean=prepared['mean'],std=prepared['std'],lifecycles=prepared['lifecycles'])
            model_file=root/'only.pt'
            torch.save(package,model_file)
            source.unlink()  # A fresh process can decode with no original sequence.
            result=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'trajectory_codec_worker.py'),
                                   'decode','--package',str(model_file),'--output-dir',str(root/'decoded'),'--device','cpu'],
                                  capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            decoded_header,frames,lifetimes=read_lifecycle_animation(root/'decoded/animation.4dgs')
            try:
                self.assertEqual(decoded_header['frameCount'],3)
                np.testing.assert_array_equal(frames['ids'],[[0,1],[0,2],[0,3]])
                np.testing.assert_array_equal(lifetimes[:,1:],life[:,:2])
                self.assertTrue(np.isfinite(frames['parameters']).all())
                for frame in range(3):
                    ids=frames[frame]['ids'].astype(np.int64)
                    identities=np.concatenate([ids*2,ids*2+1])
                    tau=np.tile(local_times(ids,frame,life,np.array(header['timesSeconds'])),(2,1))
                    with torch.no_grad():
                        expected=physical_attributes(network(package['codes'][identities],torch.from_numpy(tau)),prepared['mean'],prepared['std'])
                    np.testing.assert_allclose(frames[frame]['parameters'],expected.numpy(),atol=2e-6,rtol=2e-6)
            finally:
                frames._mmap.close();lifetimes._mmap.close()


if __name__=='__main__':
    unittest.main()
