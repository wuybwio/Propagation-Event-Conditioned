"""Station-conditioned 3-D residual regression for simulated teacher fields."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .volume_simulation import input_volume


class Block(nn.Module):
    def __init__(self, channels_in: int, channels_out: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(channels_in,channels_out,3,padding=1), nn.GroupNorm(4,channels_out),nn.SiLU(),
            nn.Conv3d(channels_out,channels_out,3,padding=1),nn.GroupNorm(4,channels_out),nn.SiLU())

    def forward(self,x):
        return self.layers(x)


class StationVolumeNet(nn.Module):
    def __init__(self, base: int = 8, residual_db: float = 60., path_samples: int = 0):
        super().__init__()
        if base < 4 or base % 4 or not math.isfinite(residual_db) or residual_db <= 0:
            raise ValueError('base must be a positive multiple of four; residual bound must be positive')
        self.residual_db = residual_db
        if not isinstance(path_samples, (int,np.integer)) or not 0 <= path_samples <= 1024:
            raise ValueError('path_samples must be an integer in [0,1024]')
        self.path_samples = int(path_samples)
        self.input_channels = 11 if path_samples else 8
        self.enc1, self.enc2 = Block(self.input_channels,base), Block(base,base*2)
        self.core = Block(base*2,base*4)
        self.dec2,self.dec1 = Block(base*6,base*2),Block(base*3,base)
        self.head=nn.Conv3d(base,1,1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self,x,coarse):
        if x.ndim != 5 or x.shape[1] != self.input_channels or coarse.shape != x[:,:1].shape or min(x.shape[2:]) < 4:
            raise ValueError(f'expected [batch,{self.input_channels},z,y,x] input and aligned coarse field')
        a=self.enc1(x)
        b=self.enc2(F.avg_pool3d(a,2))
        c=self.core(F.avg_pool3d(b,2))
        d=self.dec2(torch.cat((F.interpolate(c,size=b.shape[2:],mode='trilinear',align_corners=False),b),1))
        e=self.dec1(torch.cat((F.interpolate(d,size=a.shape[2:],mode='trilinear',align_corners=False),a),1))
        return coarse+self.residual_db*torch.tanh(self.head(e))


def coverage_boundary_dbm(threshold: float = -100., shadow_std: float = 6.) -> float:
    if not math.isfinite(threshold) or not math.isfinite(shadow_std) or shadow_std <= 0:
        raise ValueError('finite threshold and positive shadow standard deviation required')
    return threshold + shadow_std * NormalDist().inv_cdf(.9)


def training_coverage_balance(data: dict, threshold: float, shadow_std: float) -> dict:
    boundary = coverage_boundary_dbm(threshold, shadow_std)
    train_mask = np.asarray(data['split']) == 0
    targets = np.asarray(data['teacher_dbm'])[train_mask]
    air_source = np.asarray(data['air_mask'], bool)
    air = (air_source[train_mask] if air_source.ndim == 4
           else np.broadcast_to(air_source, targets.shape))
    values = targets[air]
    if not values.size or not np.isfinite(values).all():
        raise ValueError('finite nonempty training-air targets required')
    covered = int(np.sum(values >= boundary))
    counts = [int(values.size) - covered, covered]
    present = sum(count > 0 for count in counts)
    weights = [values.size / (present * count) if count else 0. for count in counts]
    return {'source': 'training_air_only', 'class_order': ['blind', 'covered'],
            'counts': counts, 'class_weights': weights, 'boundary_dbm': boundary}


def coverage_boundary_loss(prediction, target, mask, boundary_dbm: float,
                           temperature_db: float, class_weights: tuple[float, float]):
    if (prediction.shape != target.shape or mask.shape != target.shape
            or not math.isfinite(boundary_dbm) or not math.isfinite(temperature_db)
            or temperature_db <= 0 or len(class_weights) != 2
            or any(not math.isfinite(w) or w < 0 for w in class_weights)
            or sum(class_weights) <= 0):
        raise ValueError('aligned fields, positive temperature and valid class weights required')
    valid = mask > 0
    if not torch.any(valid):
        raise ValueError('nonempty air mask required')
    pred, truth = prediction[valid], target[valid]
    if not torch.isfinite(pred).all() or not torch.isfinite(truth).all():
        raise ValueError('finite air predictions and targets required')
    labels = (truth >= boundary_dbm).to(pred.dtype)
    weights = torch.where(labels > 0, class_weights[1], class_weights[0])
    # Fixed global training weights; do not rebalance each minibatch.
    logits = (pred - boundary_dbm) / temperature_db
    return (F.binary_cross_entropy_with_logits(logits, labels, reduction='none') * weights).mean()


def volume_loss(prediction, target, mask, gradient_weight: float = .05, *,
                coverage_weight: float = 0., boundary_dbm: float = -92.3106906067,
                coverage_temperature_db: float = 3.,
                coverage_class_weights: tuple[float, float] = (1., 1.)):
    if not math.isfinite(coverage_weight) or coverage_weight < 0:
        raise ValueError('coverage weight must be finite and nonnegative')
    if prediction.shape != target.shape or mask.shape != target.shape or not torch.any(mask > 0):
        raise ValueError('aligned targets and nonempty air mask required')
    mask=(mask>0).to(prediction.dtype)
    value=(F.smooth_l1_loss(prediction/20.,target/20.,reduction='none')*mask).sum()/mask.sum()
    for axis in (2,3,4):
        left=[slice(None)]*5; right=left.copy()
        left[axis]=slice(1,None); right[axis]=slice(None,-1)
        l,r=tuple(left),tuple(right)
        pair=mask[l]*mask[r]
        gradient=F.smooth_l1_loss((prediction[l]-prediction[r])/20.,
                                 (target[l]-target[r])/20.,reduction='none')
        value=value+gradient_weight*(gradient*pair).sum()/pair.sum().clamp_min(1)
    if coverage_weight:
        value = value + coverage_weight * coverage_boundary_loss(
            prediction, target, mask, boundary_dbm, coverage_temperature_db, coverage_class_weights)
    return value


def load_dataset(path: str | Path) -> dict:
    with np.load(path,allow_pickle=False) as source:
        data={key:source[key] for key in source.files}
    meta=json.loads(str(data['metadata_json'].item()))
    if meta.get('schema') != 'station_volume_v1' or meta.get('measurement_validation') is not False:
        raise ValueError('requires station_volume_v1 simulation-only dataset')
    stations = np.asarray(data['station_xyz_m'])
    if stations.ndim != 2 or stations.shape[1] != 3:
        raise ValueError('invalid station_xyz_m')
    cases = len(stations)
    axes_per_case = all(np.asarray(data[name]).ndim == 2 for name in ('x_m','y_m','z_m'))
    if axes_per_case:
        if any(np.asarray(data[name]).shape[0] != cases for name in ('x_m','y_m','z_m')):
            raise ValueError('per-case axes must have one row per station case')
        shape = (len(data['z_m'][0]), len(data['y_m'][0]), len(data['x_m'][0]))
    else:
        shape = (len(data['z_m']), len(data['y_m']), len(data['x_m']))
    expected=(cases,*shape)
    for key in ('teacher_dbm','coarse_dbm'):
        if data[key].shape != expected or not np.isfinite(data[key]).all():
            raise ValueError(f'invalid {key} field')
    air_shape = expected if np.asarray(data['air_mask']).ndim == 4 else expected[1:]
    if data['air_mask'].shape != air_shape or not data['air_mask'].any():
        raise ValueError('invalid air mask')
    if data['split'].shape != expected[:1] or set(data['split']) != {0,1,2}:
        raise ValueError('train/val/test station split required')
    if np.asarray(data['terrain_m']).ndim == 3:
        if data['terrain_m'].shape != (cases,shape[1],shape[2]):
            raise ValueError('invalid per-case terrain_m')
        if data['surface_m'].shape != data['terrain_m'].shape:
            raise ValueError('surface_m must align with terrain_m')
    elif data['terrain_m'].shape != shape[1:] or data['surface_m'].shape != shape[1:]:
        raise ValueError('invalid terrain/surface fields')
    return data


class VolumeCases(torch.utils.data.Dataset):
    def __init__(self,data: dict,split: int,path_samples: int = 0):
        self.data=data
        self.indices=np.flatnonzero(data['split']==split)
        # Geometry is static across epochs; cache only the requested split.
        self.inputs = [input_volume(data,int(i),path_samples) for i in self.indices] if path_samples else None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,index):
        i=int(self.indices[index])
        return tuple(torch.from_numpy(np.asarray(a,dtype='float32')) for a in (
            self.inputs[index] if self.inputs is not None else input_volume(self.data,i),self.data['coarse_dbm'][i,None],
            self.data['teacher_dbm'][i,None],
            (self.data['air_mask'][i] if np.asarray(self.data['air_mask']).ndim == 4
             else self.data['air_mask'])[None]))


def detection_probability(power, threshold: float = -100., shadow_std: float = 6.):
    if not math.isfinite(threshold) or not math.isfinite(shadow_std) or shadow_std <= 0:
        raise ValueError('finite threshold and positive shadow standard deviation required')
    power=np.asarray(power,float)
    return .5*(1+np.fromiter((math.erf(v) for v in ((power-threshold)/(shadow_std*math.sqrt(2))).ravel()),
                             dtype=float,count=power.size).reshape(power.shape))


def field_metrics(prediction, target, mask, threshold: float = -100., shadow_std: float = 6.) -> dict:
    if np.shape(prediction) != np.shape(target):
        raise ValueError('aligned predicted and target fields required')
    mask=np.broadcast_to(np.asarray(mask,bool),np.shape(prediction))
    a,b=np.asarray(prediction)[mask],np.asarray(target)[mask]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite predicted or teacher power in valid air')
    pa,pb=detection_probability(a,threshold,shadow_std),detection_probability(b,threshold,shadow_std)
    ca,cb=pa>=.9,pb>=.9
    tp,fp,fn,tn=(int(np.sum(v)) for v in (ca&cb,ca&~cb,~ca&cb,~ca&~cb))
    def ratio(numerator, denominator):
        return float(numerator/denominator) if denominator else None
    recall=ratio(tp,tp+fn)
    blind_recall=ratio(tn,tn+fp)
    return {'voxels':len(a),'mae_db':float(np.mean(np.abs(a-b))) if len(a) else None,
            'rmse_db':float(np.sqrt(np.mean((a-b)**2))) if len(a) else None,
            'bias_db':float(np.mean(a-b)) if len(a) else None,
            'probability_mse':float(np.mean((pa-pb)**2)) if len(a) else None,
            'coverage_iou':ratio(tp,tp+fp+fn),
            'predicted_coverage_fraction':ratio(tp+fp,len(a)),
            'teacher_coverage_fraction':ratio(tp+fn,len(a)),
            'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'coverage_precision':ratio(tp,tp+fp),'coverage_recall':recall,
            'blind_recall':blind_recall,'blind_iou':ratio(tn,tn+fp+fn),
            'false_coverage_rate':ratio(fp,tn+fp),
            'balanced_accuracy':(recall+blind_recall)/2
                if recall is not None and blind_recall is not None else None}


def diagnostic_records(data: dict, prediction: np.ndarray, teacher: np.ndarray,
                       coarse: np.ndarray, threshold: float = -100.,
                       shadow_std: float = 6.) -> list[dict]:
    """Evaluation-only strata; excess loss is NOT a geometric LOS/NLOS label."""
    if prediction.shape != teacher.shape or coarse.shape != teacher.shape:
        raise ValueError('aligned predicted, teacher and coarse fields required')
    air=np.broadcast_to(np.asarray(data['air_mask'],bool),teacher.shape)
    z_values = np.asarray(data['z_m'])
    terrain = np.asarray(data['terrain_m'])
    if z_values.ndim == 2 and terrain.ndim == 3:
        agl = z_values[:, :, None, None] - terrain[:, None, :, :]
    else:
        agl = z_values[:, None, None] - terrain[None]
    agl=np.broadcast_to(agl,teacher.shape)
    loss=coarse-teacher
    if not np.isfinite(agl[air]).all() or not np.isfinite(loss[air]).all():
        raise ValueError('nonfinite diagnostic height or loss')
    if np.any(agl[air]<0) or np.any(loss[air]<-1e-4):
        raise ValueError('requires above-ground air and attenuation-only simulation teacher')
    loss=np.maximum(loss,0.)
    groups=[('agl_m',agl,[0,30,60,90,120,160,float('inf')]),
            ('teacher_excess_loss_db',loss,[0,1,6,20,float('inf')])]
    records=[]
    for group,values,edges in groups:
        for low,high in zip(edges[:-1],edges[1:]):
            mask=air&(values>=low)&(values<high)
            records.append({'group':group,'bin':f'{low:g}_{high:g}',
                            **field_metrics(prediction,teacher,mask,threshold,shadow_std)})
    return records


def mean_residual_baselines(data, indices):
    train_indices = np.flatnonzero(np.asarray(data['split']) == 0)
    residual=data['teacher_dbm'][train_indices]-data['coarse_dbm'][train_indices]
    if len(residual)==0:
        raise ValueError('training cases required for baseline calibration')
    train_air=np.asarray(data['air_mask'],bool)
    if train_air.ndim == 3:
        train_air=np.broadcast_to(train_air,residual.shape)
    elif train_air.ndim == 4:
        train_air=train_air[train_indices]
    else:
        raise ValueError('invalid air_mask dimensions')
    scalar=float(residual[train_air].mean())
    field=residual.mean(axis=0)
    return data['coarse_dbm'][indices]+scalar,data['coarse_dbm'][indices]+field


def predict_cases(model, data, indices, device):
    outputs=[]
    model.eval()
    with torch.no_grad():
        for i in indices:
            x=torch.from_numpy(input_volume(data,int(i),model.path_samples)[None]).to(device)
            coarse=torch.from_numpy(data['coarse_dbm'][i][None,None]).to(device)
            outputs.append(model(x,coarse)[0,0].cpu().numpy())
    return np.asarray(outputs)
