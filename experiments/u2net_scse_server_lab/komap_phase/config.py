import copy
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=json.loads((ROOT/'configs/B.json').read_text())

def validate(c):
    if set(c)!=set(BASE) or c['schema_version']!=6:raise ValueError('Unsupported schema')
    for k in ('model','normalization','train','loss','evaluation','aux','paper'):
        if set(c[k])!=set(BASE[k]):raise ValueError('Missing/extra fields in '+k)
    m,t,a=c['model'],c['train'],c['aux']
    if m['architecture'] not in ('baseline','common','Unet','UnetPlusPlus','FPN','PSPNet','PAN','DeepLabV3','DeepLabV3Plus','Linknet','MAnet','Segformer','fcn','segnet','u2net','u2netp'):raise ValueError('Invalid architecture')
    if m['initialization'] not in ('imagenet','none','micronet','imagenet-micronet'):raise ValueError('Invalid initialization')
    if m['initialization'] in ('micronet','imagenet-micronet') and (m['architecture']!='baseline' or m['backbone']!='resnet101'):raise ValueError('MicroNet transfer requires B/R101')
    if m['architecture'] in ('segnet','u2net','u2netp') or m['backbone']=='rsu':
        if m['initialization']!='none':raise ValueError('This model requires scratch')
    if m['output_stride']!=32:raise ValueError('Catalogue requires OS32')
    if m['attention'] not in ('scse','none','eca','cbam','eca_cbam','apam'):raise ValueError('Invalid attention')
    if m['u2_decoder_stages'] not in ('none','all','last2') or m['u2_rsu_attention'] not in ('none','eca') or m['u2_fusion_attention'] not in ('none','cbam'):
        raise ValueError('Invalid U2 attention placement')
    if m['architecture']=='u2net':
        if m['attention'] not in ('none','scse') or ((m['attention']=='scse') != (m['u2_decoder_stages']!='none')):
            raise ValueError('U2 scSE placement/attention mismatch')
    elif any(m[k]!='none' for k in ('u2_decoder_stages','u2_rsu_attention','u2_fusion_attention')):
        raise ValueError('U2 attention fields require full U2-Net')
    if m['context'] not in ('none','aspp','lstm','cmaa','ocr','snake','frequency','weighted_skip','cross_attention'):raise ValueError('Invalid context')
    if m['skip_fusion'] not in ('none','freq_nooffset','freq_offset') or m['upsampling'] not in ('bilinear','pixelshuffle'):raise ValueError('Invalid fusion')
    if m['rrcu'] not in ('none','feature','spatial_gate','channel_gate'):raise ValueError('Invalid recurrence')
    for k in ('bem','contrast','connectivity','deep_supervision'):
        if type(m[k]) is not bool:raise ValueError('Expected boolean '+k)
    if m['contrast'] or a['contrast_weight']:raise ValueError('Contrast excluded from catalogue')
    for k in ('epochs','patch','target','stride','samples_per_epoch','batch_size'):
        if type(t[k]) is not int or t[k]<1:raise ValueError('Invalid train.'+k)
    if t['patch']!=2*t['target'] or t['patch']%32 or t['target']%32:raise ValueError('Context/center multiples32, ratio2')
    if not 0<t['stride']<=t['target'] or t['batch_size']<2 or t['samples_per_epoch']%t['batch_size']:raise ValueError('Invalid stride/batch/draws')
    if type(t['seed']) is not int or t['seed']<0 or type(t['deterministic']) is not bool:raise ValueError('Invalid seed/determinism')
    if t['workers']!=0 or t['precision']!='fp32' or t['rare_classes']!=[1,2]:raise ValueError('Fixed workers0/FP32/rare1,2')
    for k,v in [('brightness_contrast_probability',.5),('gamma_probability',.3),('noise_probability',.15)]:
        if t[k]!=v:raise ValueError('Fixed photometric recipe')
    if any(not 0<=t[k]<=1 for k in ('rare_fraction','enriched_probability')):raise ValueError('Invalid sampling')
    if t['lr']<=0 or t['weight_decay']<0 or not all(math.isfinite(t[k]) for k in ('lr','weight_decay')):raise ValueError('Invalid optimizer')
    if c['normalization']!=BASE['normalization'] or c['loss']!=BASE['loss']:raise ValueError('Fixed normalization/loss')
    if c['evaluation']['absent']!='one' or c['evaluation']['batch_size']<1 or type(c['evaluation']['final_d4']) is not bool:raise ValueError('Invalid evaluation')
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in a.values()):raise ValueError('Invalid aux value')
    if a['contact_weight']<1 or a['contact_radius']<0 or any(a[k]<0 for k in ('deep_weight','consistency_weight','ocr_weight','connectivity_weight')):raise ValueError('Invalid aux weight')
    if bool(a['deep_weight'])!=m['deep_supervision'] or bool(a['connectivity_weight'])!=m['connectivity']:raise ValueError('Head/loss mismatch')
    if a['consistency_weight'] and not m['deep_supervision']:raise ValueError('Consistency requires deep heads')
    if bool(a['ocr_weight'])!=(m['context']=='ocr'):raise ValueError('OCR head/loss mismatch')
    return c

def load_config(path):return validate(json.loads(Path(path).read_text()))
def protocol(c):
    c=copy.deepcopy(c)
    for k in ('id','description','model','aux','paper'):c.pop(k)
    return c
