"""Exercise the new joint optimizer and persistent preview protocol once on a fixture."""
import base64
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from runtime import ROOT, setup_cuda, Deformation, make_gaussians, read_config
from training import train_joint

setup_cuda()
torch.set_num_threads(2)
torch.manual_seed(7)
model = Deformation()
xyz = torch.rand((48,3), device='cuda')*.4 + torch.tensor([-.2,-.2,2.],device='cuda')
with torch.no_grad():
    xyz = model(xyz, 0)
rotation = torch.zeros((48,4),device='cuda'); rotation[:,0] = 1
g = make_gaussians(xyz, torch.rand_like(xyz), torch.full_like(xyz,.04), rotation,
                   torch.full((48,),.8,device='cuda'))
metadata = dict(width=64,height=64,K=[[60.,0.,32.],[0.,60.,32.],[0.,0.,1.]],
                normalization={'center':[0,0,0],'scale':1.},frameCount=3,fps=25,durationSeconds=.08)
rgb = np.full((3,64,64,3),100,np.uint8)
depths = np.broadcast_to(np.linspace(1,3,64,dtype=np.float32)[None,None,:],(3,64,64))
class Report:
    def check_stop(self): pass
    def __call__(self,*args,**kwargs): print(args,kwargs,flush=True)
root = ROOT/'.cache'/'dynamic_checks'; root.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir=root) as directory:
    folder = Path(directory)
    config = {**read_config(),'joint_steps':1}
    g, model = train_joint(g,model,rgb,depths,metadata,folder,folder,config,Report())
    assert torch.isfinite(g.get_xyz).all()
    request = json.dumps({'id':1,'timeSeconds':.04,'width':64,'height':64})+'\n'
    result = subprocess.run([sys.executable,'-u',str(ROOT/'workbench'/'dynamic_preview.py'),
                             '--model-dir',str(folder)],input=request,text=True,capture_output=True,
                             creationflags=subprocess.CREATE_NO_WINDOW,check=True)
    response = json.loads(result.stdout.strip().splitlines()[-1])
    assert 'error' not in response, response
    pixels = Image.open(io.BytesIO(base64.b64decode(response['png'])))
    assert pixels.size == (64,64)
    print('PASS: joint RGB/SSIM/depth backward, official Gaussian Adam, export and persistent preview protocol.',flush=True)
