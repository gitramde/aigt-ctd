import csv
import hashlib
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    require(config['target'] == 'binary', 'This baseline uses the approved binary target.')
    require(config['benign_label'] == 'Benign', 'CSE-CIC-IDS2018 uses the literal benign label Benign.')
    require(config['imputation'] == 'training_mean', 'Unsupported imputation policy')
    require(config['scaling'] == 'training_standard_scaler', 'Unsupported scaling policy')
    require(config['unknown_category'] == 'all_zero_one_hot', 'Unsupported category policy')
    require(len(config['split']['target_fractions'])==3 and
            all(v>0 for v in config['split']['target_fractions']) and
            abs(sum(config['split']['target_fractions'])-1)<1e-12, 'Split fractions must be three positive values summing to one.')
    for key in ('raw_dir', 'data_dir', 'report_dir'):
        config[key] = str((ROOT / config[key]).resolve())
    raw, data, report = (Path(config[k]) for k in ('raw_dir', 'data_dir', 'report_dir'))
    require(raw != data and raw not in data.parents and data not in raw.parents,
            'Raw and derived directories must be disjoint.')
    require(raw != report and raw not in report.parents and report not in raw.parents,
            'Reports must not overwrite raw inputs.')
    return config


def json_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def csv_write(path, rows):
    require(bool(rows), f'No rows for {path}')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def versions():
    import numpy, pandas, pyarrow, sklearn
    return dict(python=platform.python_version(), numpy=numpy.__version__, pandas=pandas.__version__,
                pyarrow=pyarrow.__version__, scikit_learn=sklearn.__version__)


def file_date(path):
    import re
    from datetime import datetime
    match = re.search(r'(\d{2}-\d{2}-\d{4})', Path(path).name)
    require(match is not None, f'No date in filename: {path}')
    return datetime.strptime(match[1], '%d-%m-%Y').date()


def target_alias(name):
    import re
    tokens = set(re.findall(r'[a-z]+', name.lower()))
    return bool(tokens & {'label', 'target', 'attack', 'class', 'benign', 'malicious', 'prediction'})
