"""Finite-loss and gradient smoke checks at the real MPS batch shape."""
import gc
import time
import torch
from mimu.research_variants import ResearchUNet, pixel_contrast, boundary_loss, cldice_loss
from mimu.loss import HybridLoss

torch.set_num_threads(4)
torch.hub.set_dir('.torch_cache/hub')
assert torch.backends.mps.is_available()
device = torch.device('mps')
x = torch.rand(4, 1, 224, 224, device=device)
y = torch.zeros(4, 224, 224, dtype=torch.long, device=device)
y[:,112:,:112] = 1
y[:,:112,112:] = 2
y[:,112:,112:] = 3
criterion = HybridLoss(pixel_loss='ce', pixel_weight=.5, dice_weight=.5,
                       class_weights=(1,1,2,1)).to(device)
for method in ['contrast', 'snake', 'boundary', 'cldice']:
    model = ResearchUNet(method, pretrained=True).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    start = time.monotonic()
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        z, f = model(x, auxiliary=True)
        if method == 'contrast':
            aux = pixel_contrast(model, f, z, y, torch.arange(4), torch.Generator().manual_seed(17))
        elif method == 'boundary':
            aux = boundary_loss(model, f, y)
        elif method == 'cldice':
            aux = cldice_loss(z, y)
        else:
            aux = z.sum()*0
        loss = criterion(z, y) + .05*aux
        assert torch.isfinite(loss)
        loss.backward()
        if method == 'snake' and step == 1:
            assert model.decoder1.snake.offset.weight.grad.abs().sum().item() > 0
        opt.step()
        print(method, step, 'loss', loss.item(), flush=True)
    print('seconds', time.monotonic()-start, flush=True)
    del model, opt, z, f, aux, loss
    gc.collect()
    torch.mps.empty_cache()
