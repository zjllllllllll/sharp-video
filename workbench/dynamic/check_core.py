"""One numerical acceptance check for inversion, blocked VJP and package reload."""
import tempfile
from pathlib import Path
import torch
from runtime import (ROOT, setup_cuda, Deformation, make_gaussians, render,
                     save_ply, load_package, read_config)

setup_cuda()
torch.manual_seed(7)
model = Deformation(16384)
points = torch.rand((48, 3), device='cuda') * .5
for t in (0., .5, 1.):
    with torch.no_grad():
        recovered = model(model(points, t), t, inverse=True)
    torch.testing.assert_close(recovered, points, atol=2e-5, rtol=2e-5)

# The memory-bounded joint-training VJP must match ordinary autograd.
model.train()
xyz = points.clone().requires_grad_()
weights = torch.randn_like(xyz)
model(xyz, .4, inverse=True).mul(weights).sum().backward()
expected_xyz = xyz.grad.clone()
expected = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
model.zero_grad(set_to_none=True)
xyz.grad = None
for start in range(0, len(xyz), 13):
    model._run(xyz[start:start+13], torch.tensor(.4, device='cuda'), True).backward(weights[start:start+13])
torch.testing.assert_close(xyz.grad, expected_xyz, atol=2e-5, rtol=2e-4)
for name, parameter in model.named_parameters():
    if name in expected:
        torch.testing.assert_close(parameter.grad, expected[name], atol=3e-4, rtol=3e-4)

model.eval()
xyz = points + torch.tensor([-.25, -.25, 2.], device='cuda')
with torch.no_grad():
    canonical = model(xyz, 0.)
rotation = torch.zeros((len(xyz), 4), device='cuda')
rotation[:, 0] = 1
g = make_gaussians(canonical, torch.rand_like(xyz), torch.full_like(xyz, .04),
                   rotation, torch.full((len(xyz),), .8, device='cuda'))
meta = {'width':64, 'height':64, 'K':[[60.,0.,32.],[0.,60.,32.],[0.,0.,1.]],
        'frameCount':3,'fps':25,'normalization':{'center':[0,0,0],'scale':1.}}
root = ROOT / '.cache' / 'dynamic_checks'
root.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(dir=root) as directory:
    folder = Path(directory)
    save_ply(g, folder/'canonical.ply')
    torch.save({'state_dict':model.state_dict(),'config':read_config(),'metadata':meta}, folder/'deformation.pt')
    loaded_g, loaded_model, loaded_meta = load_package(folder)
    for t in (0., .5, 1.):
        with torch.no_grad():
            image, depth, alpha, _ = render(g, model, meta, t)
            restored, _, _, _ = render(loaded_g, loaded_model, loaded_meta, t)
        assert torch.isfinite(image).all() and torch.isfinite(depth).all()
        torch.testing.assert_close(image, restored, atol=1e-6, rtol=1e-6)
print('PASS: inverse mapping; full vs blocked gradients; three-frame two-file reload; CUDA RGB-D renderer.', flush=True)
