"""Optional public MicroNet checkpoint compatibility check, no training."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from komap_lab.config import load_config
from komap_lab.models import SegmentationModel, load_encoder
from komap_lab.runtime import seed_everything, write_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/C11.json'))
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = load_config(args.config)
    seed_everything(42)
    model = SegmentationModel(config).eval()
    decoder = {k: v.clone() for k, v in model.decoder.state_dict().items()}
    initialization = load_encoder(model.encoder, args.weights, config['model']['initialization'])
    for key, value in decoder.items():
        torch.testing.assert_close(model.decoder.state_dict()[key], value, rtol=0, atol=0)
    with torch.inference_mode():
        output = model(torch.rand(1, 1, 224, 224))
    assert output.shape == (1, 4, 224, 224) and output.isfinite().all()
    report = {'status': 'passed', 'id': config['id'], 'initialization': initialization,
              'output': list(output.shape), 'decoder_unchanged': True, 'training_executed': False}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.report, report)
    print(report)
