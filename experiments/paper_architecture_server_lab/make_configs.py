"""Explicit reproducible architecture catalogue; never changes existing runs."""
import argparse
import copy
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CNN=['resnet18','resnet34','resnet50','resnet101','resnet152','se_resnet50','se_resnet101','se_resnet152',
     'resnext50_32x4d','resnext101_32x8d','se_resnext50_32x4d','se_resnext101_32x4d','senet154',
     *['efficientnet-b'+str(i) for i in range(8)],'densenet121','densenet161','densenet169','densenet201',
     'dpn68','dpn68b','dpn92','dpn98','dpn107','dpn131','inceptionv4','inceptionresnetv2','xception','mobilenet_v2',
     'vgg11_bn','vgg13_bn','vgg16_bn','vgg11','vgg13']
CS_CNN=['se_resnet50','se_resnet101','se_resnet152','se_resnext50_32x4d','se_resnext101_32x4d',
        'senet154','resnext101_32x8d','inceptionv4','inceptionresnetv2','densenet161','densenet201',
        'vgg13_bn','vgg16_bn','mobilenet_v2',*['efficientnet-b'+str(i) for i in range(1,6)]]
TIMM=['convnext_tiny.in12k_ft_in1k','convnext_small.in12k_ft_in1k',
      'convnextv2_tiny.fcmae_ft_in1k','convnextv2_base.fcmae_ft_in1k',
      'swin_tiny_patch4_window7_224.ms_in1k','swin_small_patch4_window7_224.ms_in1k',
      'hrnet_w18.ms_in1k','hrnet_w48.ms_in1k','mobilevit_s.cvnets_in1k']
NATIVE=['Unet','UnetPlusPlus','FPN','PSPNet','PAN','DeepLabV3','DeepLabV3Plus','Linknet','MAnet']

def slug(n):return n.replace('-','_').replace('.','_').upper()

def catalogue():
    base=json.loads((ROOT/'configs/B.json').read_text());records=[]
    def add(identifier,description,group,source,implementation='derived_module',model=None,aux=None):
        c=copy.deepcopy(base);c.update(id=identifier,description=description)
        c['model'].update(model or {});c['aux'].update(aux or {})
        c['paper']={'source':source,'implementation':implementation,'group':group};records.append(c)
    add('B',base['description'],'control','R022','maintained_baseline')
    add('B_SCRATCH','B trained from scratch','control','R022','maintained_baseline',{'initialization':'none'})
    add('B_D03','Contact CE1.5; historical negative multi-seed control','control','D03',aux={'contact_weight':1.5})
    for n in CNN:
        add('E_'+slug(n),n+' + common U-Net/scSE decoder','encoder','P03','library_encoder_common_decoder',{'architecture':'common','backbone':n})
    for n in TIMM:
        add('E_'+slug(n),n+' + common decoder; feature alignment/detail skip recorded','encoder','P02/P09/additional',
            'library_encoder_common_decoder',{'architecture':'common','backbone':'timm:'+n})
    for n in ('mit_b0','mit_b2','mit_b5'):
        add('E_'+slug(n),n+' + common decoder','encoder','P13','library_encoder_common_decoder',{'architecture':'common','backbone':n})
    for n in NATIVE:
        add('D_'+slug(n),'SMP '+n+'/ResNet101; library initialization differs from B','decoder','P03/P06/P15',
            'library_native_multiclass',{'architecture':n})
    for n in ('mit_b0','mit_b2','mit_b5'):
        add('F_SEGFORMER_'+slug(n),'Native SegFormer '+n,'native','P13','library_native_multiclass',{'architecture':'Segformer','backbone':n})
    for n in ('Unet','UnetPlusPlus','Linknet','MAnet'):
        add('PF_'+slug(n)+'_R50','PF-DiffSeg downstream '+n+'/R50 real-only (no generator)','decoder','P15',
            'library_native_real_only',{'architecture':n,'backbone':'resnet50'})
    add('F_FCN_R101','Torchvision FCN-R101/four-class head','native','P06','library_native_multiclass',{'architecture':'fcn'})
    add('F_SEGNET','VGG16-BN style pool-index SegNet, scratch','native','P06','independent_native_multiclass',{'architecture':'segnet','initialization':'none'})
    for small in (False,True):
        for deep in (False,True):
            add('F_U2'+('P' if small else '')+('_DEEP' if deep else ''),'U2NET'+('P' if small else '')+' multiclass'+(' six-head supervision' if deep else ''),
                'native','P10','upstream_native_multiclass',{'architecture':'u2netp' if small else 'u2net','initialization':'none','deep_supervision':deep},{'deep_weight':.2 if deep else 0.})
    add('E_RSU','RSU encoder adapted to baseline skip widths','encoder','P10','upstream_encoder_common_decoder',{'architecture':'common','backbone':'rsu','initialization':'none'})
    add('E_RSU_DEEP','RSU/common decoder + deep heads','module','P10',model={'architecture':'common','backbone':'rsu','initialization':'none','deep_supervision':True},aux={'deep_weight':.2})
    for n in ('none','eca','cbam','eca_cbam','apam'):
        add('M_ATTN_'+n.upper(),'Replace decoder scSE with '+n,'module','P01/P04',model={'architecture':'common','attention':n})
    for n in ('aspp','lstm','cmaa','ocr','snake','frequency','weighted_skip','cross_attention'):
        add('M_'+n.upper(),'Common decoder + '+n+' (paper-inspired)','module','P01/P08/P09/P12/P14/P16',
            model={'architecture':'common','context':n},aux={'ocr_weight':.1 if n=='ocr' else 0.})
    for n in ('feature','spatial_gate','channel_gate'):
        add('M_RRCU_'+n.upper(),'B recurrent '+n,'module','P17',model={'rrcu':n})
    add('M_FREQ_NOOFFSET','B content-adaptive low/high-pass fusion','module','P08',model={'skip_fusion':'freq_nooffset'})
    add('M_FREQ_OFFSET','Low/high-pass fusion + bounded low-res resampling','module','P08',model={'architecture':'common','skip_fusion':'freq_offset'})
    add('M_PIXELSHUFFLE','B pixel-shuffle final upsampling','module','additional',model={'upsampling':'pixelshuffle'})
    add('M_CONNECTIVITY','B four-class eight-direction connectivity auxiliary','module','P11',model={'connectivity':True},aux={'connectivity_weight':.05})
    add('M_BEM','B morphological boundary enhancement','module','additional',model={'bem':True})
    add('M_DEEP','Common decoder three-scale supervision','module','P07/P10/P14',model={'architecture':'common','deep_supervision':True},aux={'deep_weight':.2})
    add('M_CONSISTENCY','Train70 deep heads + stop-gradient scale consistency; no EMA/unlabeled data','module','P07',
        model={'architecture':'common','deep_supervision':True},aux={'deep_weight':.2,'consistency_weight':.05})
    for n in ('mit_b2','mit_b5'):
        add('M_CROSS_'+slug(n),'MiT/common decoder + adjacent-scale reduced-KV cross attention','module','P13',model={'architecture':'common','backbone':n,'context':'cross_attention'})
    add('M_HRNET_OCR','HRNet-W48/common decoder + prototype OCR','module','P09',model={'architecture':'common','backbone':'timm:hrnet_w48.ms_in1k','context':'ocr'},aux={'ocr_weight':.1})
    for n in ['resnet101']+CS_CNN:
        add('H_'+slug(n)+'_SWIN','Parallel '+n+' + Swin-T/common decoder; CS-inspired','hybrid','P02','derived_hybrid',
            {'architecture':'common','backbone':n,'hybrid_backbone':'timm:swin_tiny_patch4_window7_224.ms_in1k'})
    for n in ('micronet','imagenet-micronet'):
        add('T_R101_'+slug(n),'R101 '+n+' v1.0 with fixed ImageNet normalization','pretraining','P03',
            'maintained_encoder_weight_transfer',{'initialization':n})
    return records

def generate(output,seed=42,epochs=150,batch_size=4,smoke=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for c in catalogue():
        c['train'].update(seed=seed,epochs=epochs,batch_size=batch_size);c['evaluation']['batch_size']=batch_size
        if smoke:
            c['train'].update(patch=64,target=32,stride=16,samples_per_epoch=4,enriched_probability=0.)
            c['model']['initialization']='none'
        text=json.dumps(c,ensure_ascii=False,indent=2)+'\n';path=output/(c['id']+'.json')
        if path.exists() and path.read_text()!=text:raise ValueError('Existing config differs: '+str(path))
        if not path.exists():path.write_text(text)
    return output

def write_catalogue():
    fields=['id','group','architecture','backbone','initialization','implementation','source','description']
    with (ROOT/'EXPERIMENTS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for c in catalogue():writer.writerow({'id':c['id'],**c['paper'],**{k:c['model'][k] for k in ('architecture','backbone','initialization')},'description':c['description']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'configs');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=150);p.add_argument('--batch-size',type=int,default=4);a=p.parse_args()
    generate(a.output,a.seed,a.epochs,a.batch_size);write_catalogue()
