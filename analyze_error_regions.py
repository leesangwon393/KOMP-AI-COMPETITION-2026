"""Descriptive validation error audit and fixed baseline-selected crop panels."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage as ndi
from mimu.data import decode_mask,PALETTE

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/context_hrnet_v1/error_audit'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    folders={'baseline':ROOT/'runs/resnet34_unet_scse_ce_dice/d4_tta',
             'pixelshuffle':ROOT/'runs/upsampling_sweep_v1/pixelshuffle/d4_tta',
             'dysample':ROOT/'runs/upsampling_sweep_v1/dysample/d4_tta'}
    totals={m:dict(boundary_error=0,interior_eut_to_al=0,interior_eut_to_other=0,other_interior_error=0,total_error=0) for m in folders}
    candidates=[]
    for path in sorted((ROOT/'Data/valid/masks').glob('*.png')):
        gt=decode_mask(np.asarray(Image.open(path).convert('RGB')))
        boundary=ndi.maximum_filter(gt,size=3)!=ndi.minimum_filter(gt,size=3)
        boundary=ndi.binary_dilation(boundary,iterations=1)
        preds={m:decode_mask(np.asarray(Image.open(d/path.name).convert('RGB'))) for m,d in folders.items()}
        for m,pr in preds.items():
            err=gt!=pr
            masks={'boundary_error':err & boundary,'interior_eut_to_al':(gt==2)&(pr==3)&~boundary,
                   'interior_eut_to_other':(gt==2)&(pr!=2)&(pr!=3)&~boundary,
                   'other_interior_error':err & (gt!=2)&~boundary,'total_error':err}
            for key,mask in masks.items():totals[m][key]+=int(mask.sum())
        pr=preds['baseline']
        for y in range(0,gt.shape[0]-127,128):
            for x in range(0,gt.shape[1]-127,128):
                g,p=gt[y:y+128,x:x+128],pr[y:y+128,x:x+128]
                for kind,score in [('eut_fn',int(((g==2)&(p!=2)).sum())),('eut_fp',int(((p==2)&(g!=2)).sum())),('phase_confusion',int(((g!=p)&(g!=3)&(p!=3)).sum()))]:
                    candidates.append(dict(kind=kind,score=score,file=path.name,y=y,x=x))
    chosen=[]
    for kind in ['eut_fn','eut_fp','phase_confusion']:
        used=set()
        for row in sorted([r for r in candidates if r['kind']==kind],key=lambda r:r['score'],reverse=True):
            if row['file'] in used:continue
            chosen.append(row);used.add(row['file'])
            if len(used)==2:break
    canvas=Image.new('RGB',(5*256, len(chosen)*290+30),'white')
    draw=ImageDraw.Draw(canvas)
    for j,title in enumerate(['Input','GT','Baseline','PixelShuffle','DySample']):draw.text((j*256+8,8),title,fill='black')
    for i,row in enumerate(chosen):
        path=ROOT/'Data/valid/masks'/row['file']
        gray=Image.open(ROOT/'Data/valid/images'/row['file'].replace('_mask','_image')).convert('RGB')
        ims=[gray,Image.open(path).convert('RGB')]+[Image.open(d/row['file']).convert('RGB') for d in folders.values()]
        box=(row['x'],row['y'],row['x']+128,row['y']+128)
        for j,im in enumerate(ims):canvas.paste(im.crop(box).resize((256,256),Image.Resampling.NEAREST),(j*256,30+i*290))
        draw.text((5,288+i*290),f"{row['kind']} {row['file']} x={row['x']} y={row['y']}",fill='black')
    canvas.save(OUT/'error_panels.png')
    (OUT/'audit.json').write_text(json.dumps({'totals':totals,'selected_crops':chosen,'selection':'Top baseline error pixel count in 128px nonoverlapping grid; 2 distinct images/category. Descriptive only, not a representative sample. Boundary is a GT label transition neighborhood, categories cannot establish physical causes.'},indent=2))
    print(json.dumps(totals,indent=2))


if __name__=='__main__':main()
