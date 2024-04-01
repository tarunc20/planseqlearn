import os
from doodad.slurm.slurm_util import wrap_command_with_sbatch_matrix, SlurmConfigMatrix
import time
import datetime
import os
import subprocess
import traceback

import dateutil.tz
import sys
import hydra

if __name__ == "__main__":
    # generate singularity cmds 
    SINGULARITY_PRE_CMDS = [
        "export MUJOCO_GL='egl'",
        "export MKL_THREADING_LAYER=GNU",
        "export D4RL_SUPPRESS_IMPORT_ERROR='1'",
    ]
    slurm_config = SlurmConfigMatrix(
        partition=os.environ["SLURM_PARTITION"],
        time="72:00:00",
        n_gpus=1,
        n_cpus_per_gpu=20,
        mem="62g",
        extra_flags=os.environ["SLURM_FLAGS"],  # throw out non-RTX
    )
    experiment_subdir = "generated_videos_mopa"
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime(f"%Y_%m_%d_%H_%M_%S_{now.microsecond}")
    cwd = os.getcwd()
    exp_id = "15"
    log_dir = os.path.join(
        cwd, f"exp_local/{experiment_subdir}/%s_%s" % (exp_id, timestamp)
    )
    os.makedirs(log_dir)
    python_cmd = subprocess.check_output("which python", shell=True).decode("utf-8")[
        :-1
    ]
    logdir = log_dir
    singularity_pre_cmds = " && ".join(SINGULARITY_PRE_CMDS)
    script_name = "generate_video_from_states_non_robosuite.py"
    command_line_args = ["--env_name=SawyerLiftObstacle-v0", "--camera_name=wrist", "--suite=mopa"]
    command = " ".join((python_cmd, script_name, "", *command_line_args))
    slurm_cmd = wrap_command_with_sbatch_matrix(
        f'{os.environ["LAUNCH_SINGULARITY"]} "'
        + singularity_pre_cmds
        + " && source ~/.bashrc && mamba activate planseqlearn && "
        + command
        + '"',
        slurm_config,
        logdir,
    )
    print(slurm_cmd)