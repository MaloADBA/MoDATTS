import argparse
import os
import shutil
import subprocess
import re

def get_args():
    parser = argparse.ArgumentParser(description=''
        'Get the largest validation Dice score from `val_log.txt` in every '
        'experiment directory.')
    parser.add_argument('experiment_paths', type=str, nargs='+')
    parser.add_argument('--copy_images', type=str, default=None,
                        help='Copy the image outputs corresponding to the '
                             'best epoch into this directory.')
    parser.add_argument('--check_job_status', action='store_true',
                        help='If set, will print the status of every job.')
    parser.add_argument('--training', action='store_true',
                        help='If set, also report the training Dice in the '
                             'same epoch.')
    args= parser.parse_args()
    return args


def get_best_score(f):
    lines = f.readlines()
    best_score = 0
    best_l_num = 0
    l_num = 0
    regex = re.compile('(?<=dice\=)-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?')
    for l in lines:
        result = regex.search(l)
        if result:
            l_num += 1
            score = float(l[result.start():result.end()])
            if score < best_score:
                best_score = score
                best_l_num = l_num
    return best_score, best_l_num, len(lines)


def parse_train_dice(f, line_number):
    line = f.readlines()[line_number]
    regex = re.compile('(?<=dice\=)-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?')
    result = regex.search(line)
    if result:
        score = float(line[result.start():result.end()])
        return score
    return None


def copy_images(copy_from, copy_to, epoch):
    from_img_dir = os.path.join(copy_from, 'images')
    to_img_dir = os.path.join(copy_to, os.path.basename(copy_from))
    if not os.path.exists(to_img_dir):
        os.makedirs(to_img_dir)
    for fn in os.listdir(from_img_dir):
        if fn.split('_')[0]==str(epoch-1):
            from_path = os.path.join(from_img_dir, fn)
            to_path = os.path.join(to_img_dir, fn)
            shutil.copyfile(from_path, to_path)


def get_status_canada(job_id):
    status = None
    sacct_output = subprocess.check_output(['sacct', '-j', job_id],
                                           encoding='utf-8')
    for line in sacct_output.split('\n'):
        if re.search('\s{}\s'.format(job_id), line):
            # Status is in the last column : last word in the line.
            status = re.search('\s(\w+)\s*$', line).group(0).strip(' \t\n')
            break
    return status


def find_latest_lock_file(path):
    if not os.path.isdir(path):
        return None
    biggest_n = None
    for fn in os.listdir(path):
        match = re.search('^lock.(\d*)$', fn)
        if match:
            n = int(match.group(1))
            if biggest_n is None:
                biggest_n = 0
            if n > biggest_n:
                biggest_n = n
    if biggest_n is None:
        return None
    return os.path.join(path, f'lock.{biggest_n}')


if __name__=='__main__':
    args = get_args()
    for path in args.experiment_paths:
        if not os.path.isdir(path):
            continue
        
        # Check status of job.
        status_str = ''
        if args.check_job_status:
            status = '?'
            lock_path = find_latest_lock_file(path)
            with open(lock_path, 'r') as lock:
                pid = int(lock.readline())
                match = re.search("(?<=Submitted batch job )[0-9].*[0-9]$",
                                lock.readline())
            if match:
                job_id = match.group(0)
                status = get_status_canada(job_id)
        
        # Get greatest validation score.
        try:
            f = open(os.path.join(path, 'val_log.txt'), 'r')
            best_score, line_number, total_lines = get_best_score(f)
            if args.check_job_status:
                status_str = f'{status[0]} [{total_lines}] : '
            score_str = f'line {line_number} : {best_score}'
        except:
            if args.check_job_status:
                status_str = f'{status[0]} [0] : '
            score_str = 'None'
        
        # Get associated training score.
        if args.training:
            try:
                f = open(os.path.join(path, 'log.txt'), 'r')
                train_score = parse_train_dice(f, line_number)
                score_str = f'{score_str} {train_score}'
            except:
                score_str = f'{score_str} None'
        
        # Print.
        print(f'{status_str}{path} : {score_str}')
        
        # Copy images corresponding to the best epoch according to validation.
        if args.copy_images is not None:
            if os.path.exists(os.path.join(path, 'images')):
                copy_images(copy_from=path,
                            copy_to=args.copy_images,
                            epoch=line_number)
