import argparse
import os
import re
import subprocess

def get_args():
    parser = argparse.ArgumentParser(description="Run scancel on all jobs "
        "associated with each experiment path provided as args.")
    parser.add_argument('experiment_paths', type=str, nargs='+')
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = get_args()
    for path in args.experiment_paths:
        print(path)
        if not os.path.exists(path):
            raise ValueError("path not found: {}".format(path))
        regex = re.compile('lock\.\d+')
        lock_files = list(filter(lambda fn : regex.search(fn),
                                sorted(os.listdir(path))))
        job_id_list = []
        for fn in lock_files:
            lock_path = os.path.join(path, fn)
            with open(lock_path, 'r') as f:
                pid = f.readline()
                job_string = f.readline()   # Second line.
            match = re.search("(?<=Submitted batch job )[0-9].*[0-9]$", job_string)
            if match:
                job_id = match.group(0)
                job_id_list.append(job_id)
        for job_id in job_id_list:
            print("scancel {}".format(job_id))
            subprocess.run(["scancel", str(job_id)])
