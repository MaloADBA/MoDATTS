import argparse
import os
import warnings
import torch

def get_parser():
    parser = argparse.ArgumentParser(description="Recover experiment setup "
                                     "files from saved experiment states.")
    parser.add_argument('experiment_dirs', type=str, nargs='+')
    parser.add_argument('save_to', type=str)
    return parser


if __name__=='__main__':
    parser = get_parser()
    args = parser.parse_args()
    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to)
    for d in args.experiment_dirs:
        if not os.path.exists(d):
            warnings.warn("Path {} doesn't exist -- skipped.".format(d))
            continue
        if not os.path.isdir(d):
            warnings.warn("Path {} is not a directory -- skipped.".format(d))
            continue
        
        # Pick out a state file.
        saved_dict = None
        for fn in os.listdir(d):
            if 'state_dict' in fn and fn.endswith('.pth'):
                try:
                    saved_dict = torch.load(os.path.join(d, fn))
                except EOFError:
                    warnings.warn("Failed to read {}."
                                  "".format(os.path.join(d, fn)))
                    continue
        if saved_dict is None:
            warnings.warn("No usable state files found in {} "
                          "-- skipped.".format(d))
            continue
        
        # Open state file.
        config = saved_dict['model_as_str']
        save_fn = os.path.split(d)[-1]
        with open(os.path.join(args.save_to, save_fn), 'wt') as f:
            f.write(config)
            f.close()
        print("Recovered {}.".format(save_fn))
