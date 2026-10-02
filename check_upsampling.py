import torch
from torch.nn import functional as F
from mimu.upsampling_variants import DySampleLP, sample_pixels, UpsamplingUNet
from mimu.resnet34_unet import ResNet34UNet
from upsampling_sweep import common_digest

torch.set_num_threads(2)
torch.manual_seed(5)
x=torch.randn(2,3,4,5,dtype=torch.double,requires_grad=True)
coords=(torch.rand(2,8,10,2,dtype=torch.double)*8-2).requires_grad_()
grid=2*(coords+.5)/torch.tensor([5,4],dtype=torch.double)-1
a=sample_pixels(x,coords)
b=F.grid_sample(x,grid,padding_mode='border',align_corners=False)
assert torch.allclose(a,b,atol=1e-10)
ga=torch.autograd.grad(a.square().sum(),(x,coords),retain_graph=True)
gb=torch.autograd.grad(b.square().sum(),(x,coords))
assert all(torch.allclose(u,v,atol=1e-9) for u,v in zip(ga,gb))
d=DySampleLP(8,4)
with torch.no_grad():d.offset.weight.zero_();d.offset.bias.zero_()
x=torch.rand(2,8,5,7)
assert torch.allclose(d(x),F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False),atol=1e-6)
torch.manual_seed(42)
base=ResNet34UNet(pretrained=False).eval()
expected=common_digest(base)
rng=torch.get_rng_state()
x=torch.rand(2,1,64,64)
for name in ['fullres_skip','pixelshuffle','dysample']:
    torch.manual_seed(42)
    model=UpsamplingUNet(name,pretrained=False).eval()
    assert common_digest(model)==expected
    assert torch.equal(torch.get_rng_state(),rng)
    y=model(x)
    assert y.shape==(2,4,64,64)
    if name=='fullres_skip':assert torch.allclose(y,base(x),atol=1e-6)
    y.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if name=='dysample':assert model.upsampler.offset.weight.grad.abs().sum()>0
    print(name,'forward/backward, common initialization and RNG: OK')
print('Bilinear output/coordinate gradients and zero-offset DySample oracle: OK')
