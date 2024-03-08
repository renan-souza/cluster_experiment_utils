import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from cluster_experiment_utils.cluster_utils.base_cluster_utils import BaseClusterUtils


def parse_args():
    parser = argparse.ArgumentParser(description="Job submission script")

    parser.add_argument("--conf", help="Yaml configuration file", required=True)

    if len(sys.argv) == 1:
        parser.print_help()

    return parser.parse_args()


def main(exp_conf, conf_file, varying_param_key):
    skip = exp_conf["varying_params"][varying_param_key].get("skip", False)
    if skip:
        print(f"Skiping {varying_param_key}")
        return

    proj_id = exp_conf["static_params"]["project_id"]
    queue = exp_conf["static_params"]["queue"]
    job_name = exp_conf["static_params"]["job_name"]
    proj_dir = exp_conf["static_params"]["proj_dir"]
    conda_env = exp_conf["static_params"]["conda_env"]
    exec_script = exp_conf["static_params"]["job_execution_script"]
    
    nnodes = exp_conf["varying_params"][varying_param_key]["nnodes"]
    wall_time = exp_conf["varying_params"][varying_param_key]["wall_time"]
    custom_attributes = exp_conf["static_params"]["custom_batch_job_attributes"]
    
    os.makedirs(proj_dir, exist_ok=True)

    now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
    my_job_id = f"job_{job_name}_{varying_param_key}_{now}"

    job_dir = os.path.join(proj_dir, "exps", my_job_id)
    os.makedirs(job_dir, exist_ok=True)

    main_cmd = f"python {exec_script} --conf {conf_file} --varying_param_key {varying_param_key} --my-job-id {my_job_id}"
    #python /lustre/orion/stf219/scratch/souzar/cluster_experiment_utils/executors/flowcept_exp_executor/run_dask_job.py --conf /lustre/orion/stf219/scratch/souzar/cluster_experiment_utils/exp_dir/conf/llm_exp_params.yaml --varying_param_key small_test --my-job-id my_jobao_1

    # I replaced source activate with conda activate in frontier
#     cmd = f"""
#     module load python && \
#     conda activate {conda_env} && \
#     which python && \
#     {main_cmd} && \
#     echo Good job! && \
#     exit 0
# """.strip()
    # BSUB -alloc_flags "gpumps maximizegpfs NVME smt4" #4 simultaneous multithreading (smt) is the default on Summit
    # SBATCH --ntasks-per-node=1
    cluster_utils = BaseClusterUtils.get_instance()

    cluster_utils.submit_batch_job(
        main_cmd,
        proj_id,
        wall_time=wall_time,
        job_name=varying_param_key,
        queue_name=queue,
        custom_attributes=custom_attributes,
        stdout=f"{job_dir}/job_out.log",
        stderr=f"{job_dir}/job_err.log",
        node_count=nnodes,
    )


if __name__ == "__main__":
    args = parse_args()
    exp_conf = OmegaConf.load(Path(args.conf))

    for varying_param_key in exp_conf["varying_params"]:
        main(
            exp_conf=exp_conf, conf_file=args.conf, varying_param_key=varying_param_key
        )

    sys.exit(0)
