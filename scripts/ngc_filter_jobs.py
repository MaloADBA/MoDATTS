import argparse
import subprocess

def get_args():
    parser = argparse.ArgumentParser(description="Filter NGC job list.")
    parser.add_argument('--status', type=str, default='RUNNING',
                        choices=['RUNNING', 'FAILED', 'KILLED_BY_USER',
                                 'KILLED_BY_SYSTEM', 'FINISHED_SUCCESS',
                                 'TASK_LOST'],
                        help="Show only jobs with this status.")
    parser.add_argument('--show_name', action='store_true',
                        help="Show the job name.")
    parser.add_argument('--exclude', type=str, default=None,
                        help="File containing a list of job IDs to exclude.")
    parser.add_argument('--days', type=int, default=365,
                        help="Query jobs up to this many days old.")
    args = parser.parse_args()
    return args


def scrub(batch_list, status, show_name=False, exclude=None):
    # List of jobs to exclude.
    job_id_exclude = []
    if exclude is not None:
        with open(exclude, 'r') as f:
            job_id_exclude = [line.strip() for line in f]
    # Collect jobs.
    job_id_list = []
    job_name_list = []
    for line in batch_list.split('\n'):
        segments = line.replace(' ','').split('|')
        if len(segments)!=7:
            # Not a useful line.
            continue
        job_id = segments[1]
        job_name = segments[2]
        if len(job_id)==0:
            # No job ID on this line; skip.
            continue
        try:
            int(job_id)
        except ValueError:
            # Not a number.
            continue
        if segments[3]==status and job_id not in job_id_exclude:
            job_id_list.append(job_id)
            job_name_list.append(job_name)
    return job_id_list, job_name_list



if __name__=='__main__':
    args = get_args()
    output = subprocess.check_output(["ngc batch list --duration {}D"
                                      "".format(args.days)],
                                     shell=True)
    job_id_list, job_name_list = scrub(output.decode('utf-8'),
                                       args.status,
                                       args.show_name,
                                       args.exclude)
    if args.show_name:
        lines = ["{} {}".format(job_id, job_name)
                 for job_id, job_name in zip(job_id_list, job_name_list)]
    else:
        lines = job_id_list
    print("\n".join(lines))
