"""Match central-supervision training: 448 input, keep central224 predictions."""
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from mimu.context_hrnet import ContextUNet
from mimu.data import FullResolutionDataset,_patch_origins,PALETTE
from mimu.metrics import confusion_matrix,summarize,class_iou

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/context_hrnet_v1/halo_eval'


@torch.inference_mode()
def predict(model,image,device,tta=True):
    h,w=image.shape
    padded=np.pad(image,112,mode='reflect')
    origins=_patch_origins(h,w,224,112)
    sums=torch.zeros(2,4,h,w)
    coverage=torch.zeros(h,w)
    for start in range(0,len(origins),4):
        loc=origins[start:start+4]
        x=torch.stack([torch.from_numpy(padded[y:y+448,z:z+448].copy()).float()[None]/255 for y,z in loc]).to(device)
        combined=torch.zeros(len(loc),4,224,224)
        original=None
        for k in range(4) if tta else range(1):
            for flip in (False,True) if tta else (False,):
                view=x.rot90(k,(-2,-1))
                if flip:view=view.flip(-1)
                prob=model(view).softmax(1)
                if flip:prob=prob.flip(-1)
                prob=prob.rot90(-k,(-2,-1))[...,112:336,112:336].cpu()
                combined+=prob/(8 if tta else 1)
                if k==0 and not flip:original=prob
        for j,(y,z) in enumerate(loc):
            sums[0,:,y:y+224,z:z+224]+=original[j]
            sums[1,:,y:y+224,z:z+224]+=combined[j]
            coverage[y:y+224,z:z+224]+=1
    return sums/coverage.clamp_min(1)[None,None]


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    ds=FullResolutionDataset(ROOT/'Data','valid')
    checkpoints=[('baseline_best',ROOT/'runs/resnet34_unet_scse_ce_dice/best.pt'),
                 ('context_best224',ROOT/'runs/context_hrnet_v1/context448/best.pt'),
                 ('context_last100',ROOT/'runs/context_hrnet_v1/context448/last.pt')]
    rows=[]
    for name,path in checkpoints:
        folder=OUT/name;folder.mkdir(exist_ok=True)
        result=folder/'metrics.json'
        if result.exists():rows.append(json.loads(result.read_text()));continue
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        model=ContextUNet(False)
        model.load_state_dict(checkpoint['model']);model.pretrained=True;model.to('mps').eval()
        matrices=[[],[]]
        for i,image in enumerate(ds.images):
            probs=predict(model,image,'mps')
            for mode in range(2):
                pred=probs[mode].argmax(0).numpy().astype(np.uint8)
                matrices[mode].append(confusion_matrix(ds.original_labels[i],pred))
                dest=folder/('single' if mode==0 else 'd4');dest.mkdir(exist_ok=True)
                Image.fromarray(PALETTE[pred]).save(dest/ds.paths[i].name.replace('_image','_mask'))
            print('HALO',name,i+1,20,flush=True)
        record={'method':name,'checkpoint_epoch':checkpoint['epoch'],'input':448,'output_center':224,'stride':112,
                'selection':'Diagnostic correction: original best was selected with mismatched 224 inference. Last100 is fixed final checkpoint, no additional epoch search.'}
        for key,ms in zip(['single','d4'],matrices):
            record[key]=summarize(ms,'one')
            record[key]['filenames']=[p.name for p in ds.paths]
            record[key]['per_image_class_iou']=[class_iou(cm,'one').tolist() for cm in ms]
        result.write_text(json.dumps(record,indent=2));rows.append(record)
        del model,checkpoint
        torch.mps.empty_cache()
    lines=['# 중앙 supervision에 맞춘 halo 추론 보정', '',
           '448 입력으로 중앙224만 학습한 모델을 224 입력으로 검증한 기존 평가는 학습/추론 조건이 맞지 않았습니다. 기존 224 검증으로 선택한 best와 사전 고정한 마지막100 checkpoint를 따로 평가합니다. 새로운 epoch 탐색은 하지 않습니다.',
           '동일한 halo 추론을 기준 모델에도 적용합니다. 448 입력의 중앙224만 유지하고 stride112로 확률을 병합하며 이미지 외부는 reflect padding입니다.', '',
           '| 모델 | Epoch | 일반 mIoU % | D4 mIoU % | D4 Eut IoU % |','|---|---:|---:|---:|---:|']
    for row in rows:lines.append(f'| {row["method"]} | {row["checkpoint_epoch"]} | {row["single"]["miou"]*100:.4f} | {row["d4"]["miou"]*100:.4f} | {row["d4"]["class_iou"][2]*100:.4f} |')
    (OUT/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
