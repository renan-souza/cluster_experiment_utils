import json
import os
import redis
import time
from pathlib import Path
from typing import Any, Dict
import getpass


from omegaconf import OmegaConf, DictConfig

from cluster_experiment_utils.utils import (
    printed_sleep,
    run_cmd,
    replace_var_mapping_in_str,
)


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
    conf_str = replace_var_mapping_in_str(conf_str, variable_mapping)
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
            "user": getpass.getuser(),
            "campaign_id": exp_conf.static_params.campaign_id,
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

    flowcept_userdir_path = os.path.expanduser(f"~/.flowcept")
    os.makedirs(flowcept_userdir_path, exist_ok=True)
    destination_file = os.path.join(flowcept_userdir_path, "settings.yaml")
    OmegaConf.save(new_settings, Path(destination_file))

    print(f"Saved new {flowcept_settings_path}")
    print(f"Saved new {destination_file}")
    print(repr(new_settings))
    os.environ["FLOWCEPT_SETTINGS_PATH"] = flowcept_settings_path
    return new_settings


def kill_dbs(db_host, should_start_mongo):
    print("Killing redis...")
    if should_start_mongo:
        print("Killing mongo...")
        run_cmd(f"ssh {db_host} pkill -f mongod &")
    run_cmd(f"ssh {db_host} pkill -9 -f redis-server &")
    printed_sleep(5)


def start_mongo(db_host, mongo_start_cmd, rep_dir):
    print("Starting MongoDB...")
    mongo_data_dir = os.path.join(rep_dir, "mongo_data")
    mongo_data_dir_db = os.path.join(mongo_data_dir, "db")
    os.makedirs(mongo_data_dir_db, exist_ok=True)
    mongo_log = os.path.join(mongo_data_dir, "mongo.log")
    open(mongo_log, "w").close()

    variable_mapping = {"MONGO_DATA": mongo_data_dir_db, "MONGO_LOG": mongo_log}
    mongo_start_cmd = replace_var_mapping_in_str(mongo_start_cmd, variable_mapping)

    if "local" in db_host:
        run_cmd(f"{mongo_start_cmd} &")
    else:
        run_cmd(f"ssh {db_host} {mongo_start_cmd} & ")
    printed_sleep(5)
    print("Mongo UP!")


def start_redis(exp_conf, flowcept_settings):
    redis_start_cmd = exp_conf.static_params.redis_start_command
    db_host = flowcept_settings.main_redis.host
    db_port = flowcept_settings.main_redis.port

    print("Starting Redis")
    run_cmd(f"ssh {db_host} {redis_start_cmd} &")
    printed_sleep(2)

    start_time = time.time()
    trials = 0
    success = False
    max_trials = 10
    check_interval = 7
    max_duration = 30

    while trials < max_trials and time.time() - start_time < max_duration:
        try:
            r = redis.StrictRedis(host=db_host, port=db_port, decode_responses=True)
            r.ping()
            success = True
            break
        except redis.ConnectionError as e:
            print(
                f"Trial {trials + 1}: Redis connection failed. We're sending the kill cmd and Retrying in {check_interval} seconds."
            )
            run_cmd(f"ssh {db_host} pkill -9 -f redis-server &")
            time.sleep(check_interval)
            trials += 1

    if success:
        print("Done starting Redis.")
    else:
        print(f"Unable to establish a connection to Redis after {max_trials} trials.")


def test_data_and_persist(rep_dir, wf_result, job_output):
    if wf_result is None:
        print("We couldn't get wf_result, so we can't persist the wf result")
        return
    from flowcept import DBAPI
    from flowcept import WorkflowObject
    from flowcept import TaskQueryAPI

    api = TaskQueryAPI()

    wf_id = wf_result.get("workflow_id")
    docs = api.query(filter={"workflow_id": wf_id})

    if len(docs):
        print("Great! Found docs with the workflow_id in the tasks collection.")

    db_api = DBAPI()
    wfobj = WorkflowObject()
    wfobj.workflow_id = wf_id
    wfobj.custom_metadata = {"workflow_result": wf_result, "job_output": job_output}
    db_api.insert_or_update_workflow(wfobj)

    # Retrieving full wf info
    wfobj = db_api.get_workflow(wf_id)

    dump_file = os.path.join(rep_dir, f"db_dump_tasks_wf_{wf_id}.zip")
    db_api.dump_to_file(
        filter={"workflow_id": wf_id}, output_file=dump_file, should_zip=True
    )
    wf_obj_file = os.path.join(rep_dir, f"wf_obj_{wf_id}.json")
    with open(wf_obj_file, "w") as json_file:
        json.dump(wfobj.to_dict(), json_file, indent=2)

    print(f"Saved files {dump_file} and {wf_obj_file}.")
