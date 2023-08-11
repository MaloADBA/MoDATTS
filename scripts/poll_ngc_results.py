import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from distutils.dir_util import copy_tree
import filecmp
from functools import partial
import multiprocessing
import os
import shutil
import subprocess
import sys
import time

def get_args():
    parser = argparse.ArgumentParser(description="NGC results polling.")
    parser.add_argument('--working_dir', type=str, default='./.ngc_meta',
                        help="temporary data and record of ended jobs")
    parser.add_argument('--target_dir', type=str, default='./experiments',
                        help="transfer results here")
    parser.add_argument('--poll_every', type=int, default=240, help="minutes")
    parser.add_argument('--n_workers', type=int, default=32)
    parser.add_argument('--max_download_attempts', type=int, default=5)
    parser.add_argument('--days', type=int, default=365,
                        help="max job age")
    args = parser.parse_args()
    return args


def scrub(batch_list):
    jobs = OrderedDict()
    for line in batch_list.split('\n'):
        segments = line.replace(' ','').split('|')
        if len(segments)!=7:
            # Not a useful line.
            continue
        job_id = segments[1]
        if len(job_id)==0:
            # No job ID on this line; skip.
            continue
        try:
            int(job_id)
        except ValueError:
            # Not a number.
            continue
        status = segments[3]
        jobs[job_id] = status
    return jobs


def is_equal(dir1, dir2):
    # https://stackoverflow.com/a/6681395
    dirs_cmp = filecmp.dircmp(dir1, dir2)
    if len(dirs_cmp.left_only)>0 or len(dirs_cmp.right_only)>0 or \
        len(dirs_cmp.funny_files)>0:
        return False
    (_, mismatch, errors) =  filecmp.cmpfiles(
        dir1, dir2, dirs_cmp.common_files, shallow=False)
    if len(mismatch)>0 or len(errors)>0:
        return False
    for common_dir in dirs_cmp.common_dirs:
        new_dir1 = os.path.join(dir1, common_dir)
        new_dir2 = os.path.join(dir2, common_dir)
        if not is_equal(new_dir1, new_dir2):
            return False
    return True


def get_deepest_dir(root_dir):
    n_directories = 0
    n_files = 0
    _dir = None
    for fn in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, fn)):
            n_directories += 1
            _dir = os.path.join(root_dir, fn)
        elif fn!='joblog.log':  # Ignore NGC job log.
            n_files += 1
    if n_directories==1 and n_files==0:
        sub_dir = get_deepest_dir(_dir)
        sub_dir = sub_dir.replace(root_dir+'/', '')
        return os.path.join(root_dir, sub_dir)
    return root_dir


def download(job_id, working_dir, target_dir):
    # Silent.
    try:
        job_dir = os.path.join(working_dir, "jobs")
        subprocess.check_output(["cd {}; "
                                 "ngc result download {}"
                                 "".format(job_dir, job_id)], 
                                shell=True)
    except:
        # Failed. Delete files.
        if os.path.exists(os.path.join(working_dir, "jobs", job_id)):
            shutil.rmtree(os.path.join(working_dir, "jobs", job_id))


def update(working_dir, target_dir, local=False, n_workers=32,
           max_download_attempts=5, days=365):
    # Read out all job IDs on NGC.
    output = subprocess.check_output(["ngc batch list --duration {}D"
                                      "".format(days)],
                                     shell=True)
    jobs = scrub(output.decode('utf-8'))
    
    # Create working or target directories, if necessary.
    if not os.path.exists(os.path.join(working_dir, "jobs")):
        os.makedirs(os.path.join(working_dir, "jobs"))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Load list of ended jobs.
    jobs_file_path = os.path.join(working_dir, "jobs_ended")
    jobs_ended = []
    if os.path.exists(jobs_file_path):
        with open(jobs_file_path, 'r') as f:
            jobs_ended = [line.strip() for line in f]
    
    # Download all jobs that have not ended.
    jobs_to_get = [job_id for job_id, status in jobs.items()
                          if  job_id not in jobs_ended]
    print("FETCHING")
    print("\n".join(["  {}".format(job_id) for job_id in jobs_to_get]))
    remaining_jobs_to_get = [j for j in jobs_to_get]
    for attempt in range(max_download_attempts):
        # Batch download.
        if len(remaining_jobs_to_get):
            time.sleep(5)   # Wait for NGC to get its shit together.
            with multiprocessing.Pool(n_workers) as pool:
                pool.map(partial(download,
                                 working_dir=working_dir,
                                 target_dir=target_dir),
                         remaining_jobs_to_get)
        # Sometimes NGC incorrectly claims that there are no files, so check
        # whether some jobs did not download and attempt again.
        remaining_jobs_to_get = []
        for job_id in jobs_to_get:
            if not os.path.exists(os.path.join(working_dir, "jobs", job_id)):
                print("WARNING: no files downloaded for {} (attempt {}/{})"
                      "".format(job_id, attempt+1, max_download_attempts))
                remaining_jobs_to_get.append(job_id)
    
    # Check every download to see if it differs from the results in the
    # target directory, then merge any differing results into the target
    # directory. Mark each job whose results did not change as ended, to
    # avoid future downloads.
    for job_id in jobs_to_get:
        if not os.path.exists(os.path.join(working_dir, "jobs", job_id)):
            print("NO FILES DOWNLOADED FOR {}".format(job_id))
            continue
        # Identify deepest directory containing all files.
        job_dir_root = os.path.join(working_dir, "jobs", job_id)
        job_dir_deep = get_deepest_dir(job_dir_root)
        job_dir_rel  = os.path.relpath(job_dir_deep, job_dir_root)
        if job_dir_root!=job_dir_deep:
            # If there is a sole subdir in the job directory created by NGC
            # (NGC creates a directory named as the job_id), then merge treats
            # this as the root directory of the job. The (new) root directory
            # is merged with the target tree.
            job_dir_root = os.path.join(job_dir_root,
                                        os.path.split(job_dir_rel)[0])
            target_dir_deep = os.path.join(target_dir,
                                           *os.path.split(job_dir_rel))
        else:
            # The root directory is the job_id directory itself - copy it
            # directly.
            target_dir_deep = os.path.join(target_dir,
                                           os.path.basename(job_dir_root))
        # Compare download to previously downloaded results.
        if not os.path.exists(target_dir_deep):
            # Make target first, if it doesn't exist yet.
            os.makedirs(target_dir_deep)
        if is_equal(job_dir_deep, target_dir_deep):
            # No change. Stop tracking this task.
            jobs_ended.append(job_id)
            print("NO CHANGE TO {} : {} - REGISTERING JOB AS ENDED"
                  "".format(job_id, target_dir_deep))
        else:
            # Changed. Move downloaded results into target directory.
            subprocess.run(['rsync', '-rvz', '--delete',
                            job_dir_deep+'/', target_dir_deep+'/'])
            print("UPDATED RESULTS FOR {} : {}"
                  "".format(job_id, target_dir_deep))
        # Delete downloaded results in working directory.
        shutil.rmtree(os.path.join(working_dir, "jobs", job_id))
    
    # Update list of ended jobs.
    with open(jobs_file_path, 'w') as f:
        for line in jobs_ended:
            f.write("{}\n".format(line))


if __name__=='__main__':
    args = get_args()
    while True:
        update(args.working_dir, args.target_dir, n_workers=args.n_workers,
               max_download_attempts=args.max_download_attempts,
               days=args.days)
        print("====")
        for second in range(args.poll_every*60+1):
            now = time.strftime('%H:%M:%S', time.localtime(time.time()))
            sys.stdout.write("\r[{}] NEXT UPDATE IN {} MINUTES. "
                             "".format(now, args.poll_every-second//60))
            time.sleep(1)
        sys.stdout.write("\n")
    
