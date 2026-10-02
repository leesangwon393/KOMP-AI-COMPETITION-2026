import time
import torch
from mimu.context_hrnet import ContextDataset,build_model
from mimu.data import NativePatchDataset
from mimu.loss import HybridLoss
from mimu.research_variants import ResearchUNet
from research_sweep import model_digest

torch.set_num_threads(4)
torch.hub.set_dir('.torch_cache/hub')
a,b=NativePatchDataset('Data'),ContextDataset('Data')
for seed in range(20):
    torch.manual_seed(seed);im,gt,src=a[0];rng=torch.get_rng_state()
    torch.manual_seed(seed);large,truth,source=b[0]
    assert torch.equal(rng,torch.get_rng_state())
    assert source==src and torch.equal(gt,truth)
    assert torch.equal(im,large[:,112:336,112:336])
print('20 paired sampled crops, labels, augmentation and RNG: identical',flush=True)
torch.manual_seed(42);base=ResearchUNet('baseline',True);digest=model_digest(base)
torch.manual_seed(42);context=build_model('context448',True)
assert model_digest(context)==digest
del base,context
assert torch.backends.mps.is_available()
lossfn=HybridLoss(pixel_loss='ce',pixel_weight=.5,dice_weight=.5,class_weights=(1,1,2,1)).to('mps')
for name in ['context448','hrnet_w18']:
    m=build_model(name,True).to('mps').train()
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
    size=448 if name=='context448' else 224
    x=torch.rand(4,1,size,size,device='mps');y=torch.randint(4,(4,224,224),device='mps')
    start=time.monotonic()
    for step in range(2):
        opt.zero_grad();out=m(x);assert out.shape==(4,4,224,224)
        loss=lossfn(out,y);loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        opt.step()
    m.eval()
    with torch.no_grad():assert m(x).shape==(4,4,size,size)
    print(name,'MPS batch4 forward/backward OK','seconds',time.monotonic()-start,flush=True)
    del m,opt,x,y,out,loss
    torch.mps.empty_cache()
