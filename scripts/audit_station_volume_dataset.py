"""Print a construction audit without evaluating held-out RF predictions."""
import argparse
import json

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True)
    args = p.parse_args()
    with np.load(args.dataset,allow_pickle=False) as data:
        meta = json.loads(str(data['metadata_json']))
        for key in ['terrain_m','surface_m','station_xyz_m','coarse_dbm','teacher_dbm']:
            if not np.isfinite(data[key]).all():
                raise ValueError(f'nonfinite values: {key}')
        if np.any(data['teacher_dbm'] > data['coarse_dbm'] + 1e-4):
            raise ValueError('teacher power exceeds FSPL despite attenuation-only teacher')
        if np.any(data['surface_m'] < data['terrain_m'] - 1e-4):
            raise ValueError('surface below terrain')
        if not data['air_mask'].any():
            raise ValueError('empty outdoor volume')
        terrain = data['terrain_m']
        report = {'evidence':meta['evidence'],'measurement_validation':meta['measurement_validation'],
                  'terrain_source':meta.get('terrain',{}).get('source'),
                  'height_evidence':meta['height_evidence'],'building_components':meta['buildings'],
                  'source_building_ids':meta.get('building_audit',{}).get('unique_source_building_ids'),
                  'multiheight_ids_for_review':meta.get('building_audit',{}).get('multiheight_ids_for_review'),
                  'terrain_min_max_m':[float(terrain.min()),float(terrain.max())],
                  'terrain_std_m':float(terrain.std()),
                  'surface_max_m':float(data['surface_m'].max()),
                  'volume_shape_ZYX':list(data['air_mask'].shape),
                  'station_cases':len(data['station_xyz_m']),
                  'split_counts':{name:int((data['split']==i).sum())
                                  for i,name in enumerate(['train','val','test'])},
                  'z_absolute_range_m':[float(data['z_m'].min()),float(data['z_m'].max())],
                  'outdoor_voxels':int(data['air_mask'].sum()),
                  'note':'finite construction checks passed; not physical accuracy validation'}
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
