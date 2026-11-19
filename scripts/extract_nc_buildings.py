"""Extract a fresh, audited local building GeoJSON from nc_build Z-up tiles."""
import argparse
import json
from pathlib import Path

from jiangxi_radio.cesium_buildings import extract_local_buildings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tileset', required=True)
    parser.add_argument('--longitude', type=float, default=115.85)
    parser.add_argument('--latitude', type=float, default=28.68)
    parser.add_argument('--size-m', type=float, default=500)
    parser.add_argument('--same-id-policy', choices=['reject','preserve-mesh-parts'], default='reject')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f'choose a fresh output directory: {output}')
    collection, audit = extract_local_buildings(args.tileset, longitude=args.longitude,
                                                latitude=args.latitude, size_m=args.size_m,
                                                same_id_policy=args.same_id_policy)
    output.mkdir(parents=True)
    (output/'buildings.geojson').write_text(json.dumps(collection,ensure_ascii=False),encoding='utf-8')
    (output/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k != 'source_tiles'},ensure_ascii=False))


if __name__ == '__main__':
    main()
