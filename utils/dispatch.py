import argparse
from datetime import datetime
import os
import psutil
import re
import shutil
import subprocess
import sys
import time
import warnings

from natsort import natsorted

SBATCH_TIMEOUT = 60


"""
A parser that collects all cluster-specific arguments. To be merged into
an experiment's parser via `parents=[dispatch_parser]`.
"""
def dispatch_argument_parser(*args, **kwargs):
    parser = argparse.ArgumentParser(*args, **kwargs)
    parser.add_argument('--force_resume', action='store_true',
                        help="Resume a job even if it has already completed "
                             "the requested number of epochs.")
    g_sel = parser.add_argument_group('Cluster select.')
    mutex_cluster = g_sel.add_mutually_exclusive_group()
    mutex_cluster.add_argument('--dispatch_dgx', action='store_true')
    mutex_cluster.add_argument('--dispatch_ngc', action='store_true')
    mutex_cluster.add_argument('--dispatch_canada', action='store_true')
    g_dgx = parser.add_argument_group('DGX cluster')
    g_dgx.add_argument('--cluster_id', type=int, default=425)
    g_dgx.add_argument('--docker_id', type=str,
                       default="nvidian_general/"
                               "9.0-cudnn7-devel-ubuntu16.04_genseg:v2")
    g_dgx.add_argument('--gdx_gpu', type=int, default=1)
    g_dgx.add_argument('--gdx_cpu', type=int, default=2)
    g_dgx.add_argument('--gdx_mem', type=int, default=12)
    g_dgx.add_argument('--nfs_host', type=str, default="dcg-zfs-03.nvidia.com")
    g_dgx.add_argument('--nfs_path', type=str,
                       default="/export/ganloc.cosmos253/")
    g_ngc = parser.add_argument_group('NGC cluster')
    g_ngc.add_argument('--ace', type=str, default='nv-us-west-2')
    g_ngc.add_argument('--instance', type=str, default='ngcv1',
                       choices=['ngcv1', 'ngcv2', 'ngcv4', 'ngcv8'],
                       help="Number of GPUs.")
    g_ngc.add_argument('--image', type=str,
                       default="nvidian/lpr/"
                               "9.0-cudnn7-devel-ubuntu16.04_genseg:v2")
    g_ngc.add_argument('--source_id', type=str, default=None)
    g_ngc.add_argument('--dataset_id', type=str, default=None)
    g_ngc.add_argument('--workspace', type=str,
                       default='8CfEU-RDR_eu5BDfnMypNQ:/workspace')
    g_ngc.add_argument('--result', type=str, default="/results")
    g_cca = parser.add_argument_group('Compute Canada cluster')
    g_cca.add_argument('--account', type=str, default='rrg-bengioy-ad')
    g_cca.add_argument('--cca_gpu', type=int, default=1)
    g_cca.add_argument('--cca_cpu', type=int, default=8)
    g_cca.add_argument('--cca_mem', type=str, default='32G')
    g_cca.add_argument('--time', type=str, default='1-00:00',
                       help="Max run time (DD-HH:MM). Shorter times get "
                            "higher priority.")
    g_cca.add_argument('--copy_local', action='store_true',
                       help="Copy \'data\' to the local scratch space.")
    return parser


"""
Given a parser that contains _all_ of an experiment's arguments (including
the cluster-specific arguments from `dispatch_parser`), as well as a run()
method, run the experiment on the specified cluster or locally if no cluster
is specified.

NOTE: args must contain `path`.
"""
def dispatch(parser, run):
    # Get arguments.
    args = parser.parse_args()
    assert hasattr(args, 'path')
    
    # If resuming, check whether the requested number of epochs has already
    # been done. If so, don't resume unless `--force_resume` is used.
    if os.path.exists(os.path.join(args.path, "args.txt")):
        state_file_list = natsorted([fn for fn in os.listdir(args.path)
                                     if fn.startswith('state_dict_')
                                     and fn.endswith('.pth')])
        epoch = 0
        if len(state_file_list):
            state_file = state_file_list[-1]
            epoch = re.search('(?<=state_dict_)\d+(?=.pth)', state_file).group(0)
            epoch = int(epoch)
        if epoch >= args.epochs and not args.force_resume:
            print("WARNING: aborting dispatch since {} epochs already "
                  "completed ({}). To override, use `--force_resume`"
                  "".format(args.epochs, args.path))
            return
    
    # If resuming, merge with loaded arguments (newly passed arguments
    # override loaded arguments).
    if os.path.exists(os.path.join(args.path, "args.txt")):
        with open(os.path.join(args.path, "args.txt"), 'r') as f:
            saved_args = f.read().split('\n')[1:]
            args = parser.parse_args(args=saved_args)
        args = parser.parse_args(namespace=args)
    
    # Dispatch on a cluster (or run locally if none specified).
    if args.dispatch_dgx:
        _dispatch_dgx(args)
    elif args.dispatch_ngc:
        _dispatch_ngc(args)
    elif args.dispatch_canada:
        if _isrunning_canada(args.path):
            # If a job is already running on this path, exit.
            print("WARNING: aborting dispatch since there is already an "
                  "active job ({}).".format(args.path))
            return
        import daemon
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        daemon_log_file = open(os.path.join(args.path, "daemon_log.txt"), 'a')
        print("Dispatch on Compute Canada - daemonizing ({})."
              "".format(args.path))
        with daemon.DaemonContext(stdout=daemon_log_file,
                                  stderr=daemon_log_file,
                                  working_directory=args.path):
            _dispatch_canada_daemon(args)
    elif args.model_from is None and not os.path.exists(args.path):
        parser.print_help()
    else:
        if args.copy_local:
            # Copy 'data' to local scratch space.
            assert hasattr(args, 'data')
            target = os.path.join(os.environ["SLURM_TMPDIR"],
                                  os.path.basename(args.data))
            if os.path.exists(target):
                warnings.warn("{} exists - not copying".format(target))
            else:
                if os.path.isdir(args.data):
                    shutil.copytree(args.data, target)
                else:
                    shutil.copyfile(args.data, target)
                args.data = target
        run(args)


def _dispatch_dgx(args):
    pre_cmd = ("export HOME=/tmp; "
               "export ROOT=/scratch/; "
               "cd /scratch/ssl-seg-eugene; "
               "source register_submodules.sh;")
    cmd = subprocess.list2cmdline(sys.argv)       # Shell executable.
    cmd = cmd.replace(" --_dispatch_dgx", "")     # Remove recursion.
    cmd = "bash -c '{} python3 {};'".format(pre_cmd, cmd)  # Combine.
    mount_point = "/scratch"
    subprocess.run(["dgx", "job", "submit",
                    "-i", str(args.docker_id),
                    "--gpu", str(args.dgx_gpu),
                    "--cpu", str(args.dgx_cpu),
                    "--mem", str(args.dgx_mem),
                    "--clusterid", str(args.cluster_id),
                    "--volume", "{}@{}:{}".format(args.nfs_path,
                                                  args.nfs_host,
                                                  mount_point),
                    "-c", cmd])


def _dispatch_ngc(args):
    pre_cmd = ("cd /repo; "
               "source register_submodules.sh;")
    cmd = subprocess.list2cmdline(sys.argv)       # Shell executable.
    cmd = cmd.replace(" --_dispatch_ngc", "")     # Remove recursion.
    cmd = "bash -c '{} python3 {};'".format(pre_cmd, cmd)  # Combine.
    subprocess.run(["ngc", "batch", "run",
                    "--image", args.image,
                    "--ace", args.ace,
                    "--instance", args.instance,
                    "--commandline", cmd,
                    "--datasetid", args.dataset_id,
                    "--datasetid", args.source_id,
                    "--workspace", args.workspace,
                    "--result", args.result])


def _dispatch_canada(args):
    # Create a file that specifies that a job is being launched. This is to
    # prevent more than one job to be run at the same time for the same
    # experiment. This file contains the pid of the `sbatch` call; the output
    # from this call is appended to the file once it is received. For a 
    # successful call, this output should contain the slurm job ID.
    regex = re.compile('lock\.\d+')
    lock_files = list(filter(lambda fn : regex.search(fn),
                             sorted(os.listdir())))
    if len(lock_files):
        lock_num = int(re.search('\d+', lock_files[-1]).group(0))+1
    else:
        lock_num = 0
    lock_path = os.path.join(os.getcwd(), 'lock.{}'.format(lock_num))
    lock = open(lock_path, 'w')
    
    # Prepare command for sbatch.
    path_genseg_repository = os.getenv('PATH_GENSEG_REPOSITORY')
    if path_genseg_repository is None:
        path_genseg_repository = "/home/veugene/home_projects/ssl-seg-eugene"
    print(f"Using code in {path_genseg_repository}")
    path_python_daemon_wheel = os.getenv('PATH_PYTHON_DAEMON_WHEEL')
    if path_python_daemon_wheel is None:
        path_python_daemon_wheel = ("~/env/genseg/wheels/"
                                    "python_daemon-2.3.0-py2.py3-none-any.whl")
    pre_cmd = ("cd {}\n"
               "module load python/3.7 cuda cudnn scipy-stack\n"
               "virtualenv --no-download $SLURM_TMPDIR/env\n"
               "source $SLURM_TMPDIR/env/bin/activate\n"
               "pip install --no-index --upgrade pip\n"
               "pip install --no-index -r requirements.txt\n"
               "pip install {}\n"
               "source register_submodules.sh\n"
               "".format(path_genseg_repository, path_python_daemon_wheel))
    cmd = subprocess.list2cmdline(sys.argv)       # Shell executable.
    cmd = cmd.replace(" --dispatch_canada",   "") # Remove recursion.
    cmd = "#!/bin/bash\n {}\n python3 {}".format(pre_cmd, cmd)  # Combine.
    
    # Open sbatch process.
    proc = subprocess.Popen([
                    "sbatch",
                    "--account", args.account,
                    "--gres", 'gpu:{}'.format(args.cca_gpu),
                    "--cpus-per-task", str(args.cca_cpu),
                    "--mem", args.cca_mem,
                    "--time", args.time],
                   encoding='utf-8',
                   stderr=subprocess.STDOUT,
                   stdout=subprocess.PIPE,
                   stdin=subprocess.PIPE)
    
    # Record sbatch pid. Flush to make sure the file is written.
    print(proc.pid, file=lock)
    lock.flush()
    
    # Run the prepared command and wait for sbatch to complete.
    # Raise an error if it times out.
    try:
        out, err = proc.communicate(input=cmd, timeout=SBATCH_TIMEOUT)
        print(out, file=lock)
        print(out)
    except subprocess.TimeoutExpired as e:
        print("Call to `sbatch` timed out after {}s : {}"
              "".format(SBATCH_TIMEOUT, e))
        proc.terminate()
        exit()
    
    # Flush and close the lock file.
    lock.close()
    
    return out


def _dispatch_canada_daemon(args):
    # Parse time argument to seconds.
    if   args.time.count(':')==0 and args.time.count('-')==0:
        # minutes
        time_format = "%M"
    elif args.time.count(':')==0 and args.time.count('-')==1:
        # days-hours
        time_format = "%d-%H"
    elif args.time.count(':')==1 and args.time.count('-')==0:
        # minutes:seconds
        time_format = "%M:%S"
    elif args.time.count(':')==1 and args.time.count('-')==1:
        # days-hours:minutes
        time_format = "%d-%H:%M"
    elif args.time.count(':')==2 and args.time.count('-')==0:
        # hours:minutes:seconds
        time_format = "%H:%M:%S"
    elif args.time.count(':')==2 and args.time.count('-')==1:
        # days-hours:minutes:seconds
        time_format = "%d-%H:%M:%S"
    else:
        raise ValueError("Invalid `time` format ({}).".format(args.time))
    datetime_obj = datetime.strptime(args.time, time_format)
    time_seconds = ( datetime_obj
                    -datetime(datetime_obj.year, 1, 1)).total_seconds()
    if '-' in args.time:
        # Add a day if days are specified, since days count up from 1.
        time_seconds += 24*60*60
    
    # Periodically check status of job. Relaunch on TIMEOUT.
    status = 'TIMEOUT'
    while status=='TIMEOUT':
        # Launch.
        sbatch_output = _dispatch_canada(args)
        job_id = sbatch_output.split(' ')[-1].strip(' \n\t')
        try:
            int(job_id)
        except ValueError:
            raise RuntimeError("Cannot extract job ID from `sbatch` standard "
                               "output: {}".format(sbatch_output))
        
        # Wait until the job is launched before setting a timer.
        status = _get_status_canada(job_id)
        while status=='PENDING' or status is None:
            time.sleep(10)
            status = _get_status_canada(job_id)
        
        # Wait.
        time.sleep(time_seconds)
        
        # Check status.
        status = _get_status_canada(job_id)
        
        # If the job is still RUNNING, wait until the state changes.
        # 
        # The job may continue running while the cluster waits for it
        # to exit after a TERM signal (or until it eventually KILLs the job).
        while status=='RUNNING':
            time.sleep(10)
            status = _get_status_canada(job_id)


def _get_status_canada(job_id):
    status = None
    sacct_output = subprocess.check_output(["sacct", "-j", job_id],
                                           encoding='utf-8')
    for line in sacct_output.split('\n'):
        if re.search("\s{}\s".format(job_id), line):
            # Status is in the last column : last word in the line.
            status = re.search('\s(\w+)\s*$', line).group(0).strip(' \t\n')
            break
    return status


def _isrunning_canada(path):
    # Check for lock files. If they exist, check for job IDs. If there is no
    # job ID in a lock file, sbatch may still be running; check the sbatch PID.
    # If the process is still running, wait. If there is no such process,
    # assume the job has failed to launch.
    
    # If the path does not exist, nothing is running with that path.
    if not os.path.exists(path):
        return False
    
    # Helper to check if a lock is active.
    def lock_is_active(lock_path):
        with open(lock_path, 'r') as lock:
            pid = int(lock.readline())
            match = re.search("(?<=Submitted batch job )[0-9].*[0-9]$",
                              lock.readline())
        
        # Check if the sbatch process still exists.
        active = None
        try:
            proc = psutil.Process(pid)
            if not 'sbatch' in proc.name():
                raise psutil.NoSuchProcess(pid)
        except psutil.NoSuchProcess:
            active = False
        
        # If there is a job ID, check if it's active.
        if match:
            job_id = match.group(0)
            status = _get_status_canada(job_id)
            if status in ['RUNNING', 'PENDING']:
                active = True
            
        return active
    
    # Check all lock files.
    regex = re.compile('lock\.\d+')
    lock_files = list(filter(lambda fn : regex.search(fn),
                             sorted(os.listdir(path))))
    for fn in lock_files:
        active = None
        while active not in [True, False]:
            # Loop until the lock is found active or until the associated
            # sbatch process exits.
            active = lock_is_active(os.path.join(path, fn))
            if active is not None:
                break
            time.sleep(1)
        if active:
            return True
    return False
