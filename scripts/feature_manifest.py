"""Record hashes of source and produced cutoff snapshots for reproducibility."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    inputs = ['configs/benchmark.json', 'data/raw/online_retail_II.xlsx',
              'reports/generated/cohort.csv', 'reports/generated/cohort-intermittent.csv',
              'src/retailmind/product_features.py', 'src/retailmind/workbook.py',
              'scripts/materialize_metadata.py', 'scripts/refresh_description_features.py']
    outputs = sorted((ROOT / 'reports/generated/feature-snapshots').glob('*.csv'))
    if len(outputs) != 4:
        raise ValueError('Expected four validation feature snapshots; generate them first')
    manifest = {'purpose': 'Validation-cutoff feature snapshots; no training or evaluation',
                'inputs': {name: digest(ROOT/name) for name in inputs},
                'outputs': {str(p.relative_to(ROOT)): digest(p) for p in outputs},
                'limitations': ['Descriptions and prices are historical observational data', 'Customer country is not production origin', 'Pack and code-family fields require review']}
    path = ROOT/'reports/generated/feature-snapshots/manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print('Wrote', path)

if __name__ == '__main__': main()
