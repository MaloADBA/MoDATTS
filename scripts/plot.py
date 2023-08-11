import os
import argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('read_path', type=str)
    parser.add_argument('write_path', type=str)
    parser.add_argument('keyword', type=str)
    parser.add_argument('--xmin', type=float, default=None)
    parser.add_argument('--xmax', type=float, default=None)
    parser.add_argument('--ymin', type=float, default=None)
    parser.add_argument('--ymax', type=float, default=None)
    parser.add_argument('--title', type=str, default=None)
    args = parser.parse_args()
    return args


def scrub(path, keyword):
    loss_list = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            loss = None
            for l in line.split(' '):
                if l.startswith("{}=".format(keyword)):
                    loss = float(l.split('=')[-1])
                    break
            if loss is None:
                raise Exception("Keyword `{}` not found on line {}."
                                "".format(keyword, i))
            loss_list.append(loss)
    return loss_list

if __name__=='__main__':
    args = parse_args()
    val = scrub(args.read_path, args.keyword)
    plt.plot(val)
    plt.xlim([args.xmin, args.xmax])
    plt.ylim([args.ymin, args.ymax])
    if args.title is not None:
        plt.title(args.title)
    if not os.path.exists(os.path.dirname(args.write_path)):
        os.makedirs(os.path.dirname(args.write_path))
    plt.savefig(args.write_path)
