"""Write the completed baseline preparation results as a readable report."""
import argparse
import csv
import json
from pathlib import Path
from .common import ROOT, load_config, require
from .split import PARTITIONS


def table(headers,rows):
    def cell(value):
        return f'{value:,}' if isinstance(value,int) else str(value)
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(cell(value) for value in row)+' |' for row in rows])


def run(config):
    reports=Path(config['report_dir'])
    def read(name): return json.loads((reports/name).read_text())
    manifest=read('source_manifest.json'); plan=read('split_plan.json')
    state=read('preprocessing.json'); verification=read('verification.json')
    require(verification['status']=='PASS','Verification must pass before reporting completion')
    totals=manifest['totals']
    lines=['# Baseline preparation results',
           'Binary target: Benign = 0, observed attacks = 1. All original attack labels are retained as metadata. No classifier training, model evaluation, graph construction, GATv2 or AIGT-CTD implementation was performed.',
           '## Cleaning',
           table(['Measure','Records'],[(k,v) for k,v in totals.items()]),
           'The Phase 2 duplicate count includes 56 repeated-header duplicates. Removing all 59 headers first leaves 410,707 non-header duplicates. The 14 year-1970 records remain in non-temporal quarantine files and are excluded from temporal partitions.',
           'All raw SHA-256 hashes were recomputed after preparation and remain unchanged. Every raw data row is accounted for exactly once as temporally eligible, removed, or quarantined.',
           '## Frozen chronological split',
           'The complete class/file/date report was written before split selection. Whole-date candidates were ranked by training attack-class coverage and closeness to the configured 80/10/10 row proportions, subject to binary representation minimums. No shuffled or within-day stratified split was used.',
           table(['Partition','Dates','Records','Benign','Malicious','Attacks unseen in training'],[
               (part,entry['dates'][0]+' through '+entry['dates'][-1],entry['rows'],entry['class_counts'].get('Benign',0),
                entry['rows']-entry['class_counts'].get('Benign',0),', '.join(entry['unseen_attack_classes']) or 'None')
               for part in PARTITIONS for entry in [plan['partitions'][part]]]),
           'Class distribution for each partition, including zero-count classes, is saved in [partition_class_distribution.csv](partition_class_distribution.csv). Unseen attacks remain positive binary targets. These date-separated attack types prevent a claim of complete closed-set multiclass coverage.',
           'Order is by the recorded day-first timestamp, then original source row. The source supplies no timezone or AM/PM marker. No missing clock information was reconstructed.',
           '## Fitted preprocessing',
           table(['Property','Value'],[
               ('Fit partition',state['fit_partition']),('Training rows',state['training_rows']),
               ('Numeric features',len(state['numeric_columns'])),('Categorical features',len(state['categorical_columns'])),
               ('Output features',len(state['output_feature_names'])),('Protocol categories',json.dumps(state['categories'])),
               ('Numeric imputation','Mean from finite training values only'),
               ('Categorical imputation','Training mode only'),('Scaling','StandardScaler fitted on imputed training rows only'),
               ('Unknown categories','All-zero one-hot block; counted in application audit'),
               ('All-missing training features',', '.join(state['all_missing_numeric_columns']) or 'None'),
               ('Additional direct target encodings excluded',', '.join(r['column'] for r in state['excluded_target_encodings']) or 'None detected')]),
           'Label, target-named fields, timestamps and identifiers are excluded from model inputs. Training-only one-to-one target-encoding checks supplement the schema exclusions. All exclusion reasons are in [feature_exclusion_audit.csv](feature_exclusion_audit.csv). Correlation alone is not treated as proof of target leakage.',
           'The saved preprocessor was reloaded and applied unchanged to every training, validation and test row. All transformed values were finite. Feature batches are generated on demand, so no second full scaled copy consumes disk space.',
           '## Artifacts',
           table(['Artifact','Purpose'],[
               ('[cleaning_audit.csv](cleaning_audit.csv)','Each removed/excluded source row and retained duplicate reference'),
               ('[cleaning_summary.csv](cleaning_summary.csv)','Per-file row conservation'),
               ('[cleaning_cell_audit.csv](cleaning_cell_audit.csv)','Infinity-to-missing conversion and remaining missing counts'),
               ('[pre_split_class_distribution.csv](pre_split_class_distribution.csv)','Class/file/date counts before choosing a split'),
               ('[split_candidates.csv](split_candidates.csv)','Candidate boundaries, counts and deterministic ranking'),
               ('[split_plan.json](split_plan.json)','Exact dates, files, row rules, hashes and membership paths'),
               ('[preprocessing.json](preprocessing.json)','Persisted training-only imputation, categories, scaling and feature order'),
               ('[preprocessing_application_audit.csv](preprocessing_application_audit.csv)','All-partition transform counts and checksums'),
               ('[source_manifest.json](source_manifest.json)','Raw/derived paths, hashes and population counts'),
               ('[verification.json](verification.json)','PASS: source preservation, conservation, chronology and split/preprocessor integrity')]),
           'Cleaned features and exact per-row partition membership are stored under `data/baseline_v1/`, with paths and hashes in the manifests. Reproduction commands and the streaming model-input interface are documented in [BASELINE_PIPELINE.md](../../docs/BASELINE_PIPELINE.md).']
    (reports/'baseline_preparation_report.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote baseline_preparation_report.md',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/baseline.json')
    run(load_config(parser.parse_args().config))
