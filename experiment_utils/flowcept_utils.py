import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf, DictConfig

from experiment_utils.utils import printed_sleep, run_cmd


def generate_job_output(
    conf_data,
    host_allocs,
    host_counts,
    job_dir,
    my_job_id,
    proj_dir,
    python_env,
    rep_dir,
    rep_no,
    resource_usage,
    t0,
    t1,
    t_c_f,
    t_c_i,
    varying_param_key,
    wf_result,
    with_flowcept,
):
    out_job = {
        "my_job_id": my_job_id,
        "job_dir": job_dir,
        "rep_dir": rep_dir,
        "lsf_job_id": os.getenv("LSB_JOBID"),
        "lsb_hosts": host_counts,
        "varying_param_key": varying_param_key,
        "rep_no": rep_no,
        "with_flowcept": with_flowcept,
        "python_env": python_env,
        "run_end_time": datetime.utcnow().strftime("%Y-%m-%d %H-%M-%S.%f")[:-3],
        "total_time": t1 - t0,
        "client_time": t_c_f - t_c_i,
    }
    if wf_result:
        out_job["wf_result"] = wf_result
    out_job.update(conf_data["varying_params"][varying_param_key])
    out_job.update(host_allocs)
    out_job.update(conf_data["static_params"])
    out_job.update(conf_data["varying_params"][varying_param_key])
    if resource_usage is not None:
        out_job.update(resource_usage)
    print(json.dumps(out_job, indent=2))
    with open(f"{rep_dir}/out_job.json", "w") as f:
        f.write(json.dumps(out_job, indent=2) + "\n")
    with open(f"{proj_dir}/results.jsonl", "a+") as f:
        f.write(json.dumps(out_job) + "\n")


# def test_mongo(flowcept_settings, wf_result):
#     # we can implement this using flowcept db api to query the db


def omegaconf_simple_variable_mapping(
    conf: DictConfig, variable_mapping: Dict[str, Any]
) -> DictConfig:
    """
    This function does a simple search and replace of the variables, written as ${var_name} in the conf
    :param conf:
    :param variable_mapping:
    :return:
    """
    conf_str = OmegaConf.to_yaml(conf)
    for var_name in variable_mapping:
        conf_str = conf_str.replace(
            "${" + var_name + "}", str(variable_mapping[var_name])
        )
    return OmegaConf.create(conf_str)


def update_flowcept_settings(
    exp_conf: DictConfig,
    flowcept_settings: DictConfig,
    db_host,
    should_start_mongo,
    repetition_dir,
    varying_param_key,
    job_id,
):
    log_path = os.path.join(repetition_dir, "flowcept.log")
    new_settings = OmegaConf.create(flowcept_settings)
    new_settings = omegaconf_simple_variable_mapping(
        new_settings,
        variable_mapping={
            "db_host": db_host,
            "job_id": job_id,
            "log_path": log_path,
            "log_level": exp_conf.static_params.flowcept_log_level,
            "user": exp_conf.static_params.user,
            "experiment_id": exp_conf.static_params.experiment_id,
        },
    )

    new_settings.main_redis.host = db_host
    if should_start_mongo:
        new_settings.mongodb.host = db_host

    for adapter_key in new_settings.adapters:
        new_settings.adapters[adapter_key].update(
            exp_conf.varying_params[varying_param_key]["adapters"][adapter_key]
        )

    flowcept_settings_path = os.path.join(repetition_dir, "flowcept_settings.yaml")
    OmegaConf.save(new_settings, Path(flowcept_settings_path))
    print(repr(new_settings))
    os.environ["FLOWCEPT_SETTINGS_PATH"] = flowcept_settings_path
    return new_settings


def kill_dbs(db_host, should_start_mongo):
    print("Killing mongo & redis...")
    if should_start_mongo:
        run_cmd(f"ssh {db_host} pkill -9 -f mongod &")
    run_cmd(f"ssh {db_host} pkill -9 -f redis-server &")
    printed_sleep(5)


def start_mongo(db_host, mongo_image, rep_dir):
    print("Starting MongoDB...")
    mongo_data_dir = os.path.join(rep_dir, "mongo_data")
    os.makedirs(mongo_data_dir, exist_ok=True)
    os.makedirs(os.path.join(mongo_data_dir, "db"), exist_ok=True)
    run_cmd(
        f"ssh {db_host} singularity run --bind {mongo_data_dir}:/data {mongo_image} --logpath {rep_dir}/mongo.log & "
    )
    printed_sleep(10)
    print("Mongo UP!")


def start_redis(db_host, redis_image, redis_conf_file):
    print("Starting Redis")
    run_cmd(
        f"ssh {db_host} singularity run --bind {redis_conf_file}:/usr/local/etc/redis/redis.conf {redis_image} redis-server /usr/local/etc/redis/redis.conf &"
    )
    printed_sleep(2)
    print("Done starting Redis.")
