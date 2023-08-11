import argparse
import os
import pandas as pd
from utils.logging import parse_log_file


def parse_args():
    parser = argparse.ArgumentParser(description="Parse log files")
    parser.add_argument('--logname', type=str, default='')
    parser.add_argument('--outdir', type=str, default=None)
    args = parser.parse_args()
    return args

args = parse_args()

dd = parse_log_file(args.logname)
df = pd.DataFrame.from_dict(dd)
if args.outdir is None:
    out_dir = os.path.dirname(args.logname)
out_filename = os.path.join(out_dir, os.path.basename(args.logname)+".csv")
df.to_csv(out_filename)
