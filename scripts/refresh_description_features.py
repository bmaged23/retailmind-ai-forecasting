"""Reapply literal-description rules to already extracted as-of metadata."""
import csv
from pathlib import Path
from retailmind.product_features import TEXT_TERMS, description_tokens, candidate_pack_size

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'reports/generated'
    paths = [folder / name for name in ('product-features.csv', 'product-features-all.csv')]
    paths.extend(sorted((folder / 'feature-snapshots').glob('*.csv')))
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline='', encoding='utf-8') as handle:
            records = list(csv.DictReader(handle))
        for item in records:
            desc = item.get('description', '')
            item['description_available'] = int(bool(desc.strip()))
            tokens = description_tokens(desc)
            for term in TEXT_TERMS:
                item[f'text_{term.lower()}'] = int(term in tokens) if desc.strip() else None
            item['candidate_pack_size'] = candidate_pack_size(desc)
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0].keys())
            writer.writeheader(); writer.writerows(records)
        print(f'Refreshed {len(records)} records in {path.name}')

if __name__ == '__main__': main()
