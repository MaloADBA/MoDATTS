import argparse
import json
import os
import re
import subprocess

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('job_dirs', nargs='+', type=str,
                        help="will recurse into each job directory, looking "
                             "for state files")
    parser.add_argument('--epochs', type=int, required=True,
                        help="jobs which do not have state files for at least "
                             "this many epochs will be resumed and set to run "
                             "to at least this many epochs")
    parser.add_argument('--dispatch_canada', action='store_true')
    args = parser.parse_args()
    return args


def dispatch(root, args):
    with open(os.path.join(root, "args.txt")) as f:
        launcher = f.readline().strip("\n")
    cmd_args = ["python", launcher, "--path", root]
    if args.dispatch_canada:
        cmd_args.append("--dispatch_canada")
    out = subprocess.check_output(cmd_args, encoding='utf-8')
    print(out, end='')
    return out


if __name__=='__main__':
    args = get_args()
    regex = re.compile("(?<=state_dict_)\d+(?=.pth)")
    for d in args.job_dirs:
        for root, dirs, files in os.walk(d):
            dirs.sort()
            if "args.txt" not in files:
                continue    # Not an experiment directory.
            last_epoch = None
            for f in sorted(files):
                result = regex.search(f)
                if result:
                    last_epoch = int(f[result.start():result.end()])
            if last_epoch is None or last_epoch < args.epochs:
                print("RESUMING : {}".format(root))
                dispatch(root, args)
            else:
                print("DONE : {}".format(root))
