"""Reproduce the Part I cleaned data from UCI raw files, without a notebook.

Run from the project root: py -3.12 -m part2.prepare_data --download
Existing raw files are immutable and are never overwritten.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request
import zipfile

import pandas as pd

from part2 import config

UCI_URL = 'https://archive.ics.uci.edu/static/public/117/census%2Bincome%2Bkdd.zip'
COLUMNS = [
    'age', 'class_of_worker', 'detailed_industry_recode', 'detailed_occupation_recode',
    'education', 'wage_per_hour', 'enroll_in_edu_inst_last_wk', 'marital_status',
    'major_industry_code', 'major_occupation_code', 'race', 'hispanic_origin', 'sex',
    'member_of_labor_union', 'reason_for_unemployment', 'full_or_part_time_employment_status',
    'capital_gains', 'capital_losses', 'dividends_from_stocks', 'tax_filer_status',
    'region_of_previous_residence', 'state_of_previous_residence',
    'detailed_household_and_family_status', 'detailed_household_summary', 'instance_weight',
    'migration_code_change_in_msa', 'migration_code_change_in_region',
    'migration_code_move_within_region', 'live_in_this_house_1_year_ago',
    'migration_previous_residence_in_sunbelt', 'num_persons_worked_for_employer',
    'family_members_under_18', 'country_of_birth_father', 'country_of_birth_mother',
    'country_of_birth_self', 'citizenship', 'own_business_or_self_employed',
    'fill_inc_questionnaire_for_veterans_admin', 'veterans_benefits',
    'weeks_worked_in_year', 'year', 'income_class',
]
RAW_NAMES = ('census-income.data', 'census-income.test')
EXPECTED_ROWS = {'census-income.data': (199523, 196294),
                 'census-income.test': (99762, 98879)}


def download_missing(raw_dir):
    missing = {name for name in RAW_NAMES if not (raw_dir / name).exists()}
    if not missing:
        return
    print('Downloading the UCI Census-Income (KDD) archive...', flush=True)
    with urllib.request.urlopen(UCI_URL, timeout=120) as response:
        archive_bytes = response.read()
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as outer:
        archives = [n for n in outer.namelist() if Path(n).name == 'census.tar.gz']
        if len(archives) != 1:
            raise ValueError('UCI archive must contain exactly one census.tar.gz')
        with tarfile.open(fileobj=io.BytesIO(outer.read(archives[0])), mode='r:gz') as inner:
            # Read only named regular files; do not extract arbitrary archive paths.
            for member in inner.getmembers():
                name = Path(member.name).name
                if name in missing and member.isfile():
                    with inner.extractfile(member) as content:
                        with (raw_dir / name).open('xb') as destination:
                            destination.write(content.read())
                    missing.remove(name)
    if missing:
        raise ValueError(f'UCI archive is missing {sorted(missing)}')


def clean_split(path):
    frame = pd.read_csv(path, header=None, names=COLUMNS, skipinitialspace=True,
                        keep_default_na=False)
    raw_rows = len(frame)
    for column in frame.select_dtypes(include=['object', 'str']).columns:
        frame[column] = frame[column].str.strip().replace('?', 'Unknown')
    labels = set(frame[config.TARGET].unique())
    if labels != {'- 50000.', '50000+.'}:
        raise ValueError(f'Unexpected target labels: {labels}')
    # Deduplicate complete raw records, including survey weight, within each split.
    frame = frame.drop_duplicates().reset_index(drop=True)
    frame[config.TARGET] = frame[config.TARGET].eq('50000+.').astype('int64')
    if (raw_rows, len(frame)) != EXPECTED_ROWS[path.name]:
        raise ValueError(f'{path.name}: unexpected raw/cleaned counts {raw_rows}/{len(frame)}')
    if frame.isna().any().any() or frame.eq('?').any().any():
        raise ValueError('Cleaning left missing values or question-mark placeholders')
    return frame, {'raw_rows': raw_rows, 'cleaned_rows': len(frame),
                   'positive_rows': int(frame[config.TARGET].sum()),
                   'raw_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='download missing UCI raw files')
    args = parser.parse_args()
    raw_dir = config.ROOT / 'raw'
    if args.download:
        download_missing(raw_dir)
    absent = [name for name in RAW_NAMES if not (raw_dir / name).exists()]
    if absent:
        parser.error(f'Missing raw files {absent}; rerun with --download or place them in raw/')
    train, train_info = clean_split(raw_dir / RAW_NAMES[0])
    test, test_info = clean_split(raw_dir / RAW_NAMES[1])
    if list(train.columns) != list(test.columns):
        raise ValueError('Train/test schemas differ')
    config.CLEAN_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(config.CLEAN_TRAIN, index=False)
    test.to_csv(config.CLEAN_TEST, index=False)
    manifest = {'dataset': 'Census-Income (KDD)', 'source': UCI_URL,
                'train': train_info, 'test': test_info, 'columns': COLUMNS,
                'policy': 'strip strings; ? -> Unknown; deduplicate complete rows within each split; target -> 0/1'}
    path = config.CLEAN_TRAIN.parent / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
