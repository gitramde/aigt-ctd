"""Run reproducible cleaning, split selection, preprocessing and verification."""
import argparse
from pathlib import Path
from .common import ROOT, load_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/baseline.json')
    parser.add_argument('--stage',choices=['all','clean','split','preprocess','verify'],default='all')
    args=parser.parse_args()
    config=load_config(args.config)
    from . import clean, split, preprocess, verify, report
    for name,module in [('clean',clean),('split',split),('preprocess',preprocess),('verify',verify)]:
        if args.stage in ('all',name):
            module.run(config)
            if name=='verify':
                report.run(config)


if __name__=='__main__': main()
