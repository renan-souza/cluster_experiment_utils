import json
import os
from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf, DictConfig

from flowcept import TaskQueryAPI, DBAPI, WorkflowObject

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


def start_mongo(db_host, mongo_start_cmd, rep_dir):
    print("Starting MongoDB...")
    mongo_data_dir = os.path.join(rep_dir, "mongo_data")
    mongo_data_dir_db = os.path.join(mongo_data_dir, "db")
    os.makedirs(mongo_data_dir_db, exist_ok=True)
    mongo_log = os.path.join(mongo_data_dir, "mongo.log")
    open(mongo_log, "w").close()

    variable_mapping = {"MONGO_DATA": mongo_data_dir_db, "MONGO_LOG": mongo_log}
    mongo_start_cmd = replace_var_mapping_in_str(mongo_start_cmd, variable_mapping)
    run_cmd(f"ssh {db_host} {mongo_start_cmd} & ")
    printed_sleep(5)
    print("Mongo UP!")


def start_redis(db_host, redis_start_cmd):
    print("Starting Redis")
    run_cmd(f"ssh {db_host} {redis_start_cmd} &")
    printed_sleep(2)
    print("Done starting Redis.")


def test_data_and_persist(rep_dir, wf_result, job_output):
    api = TaskQueryAPI()

    wf_id = wf_result.get("workflow_id")
    docs = api.query(filter={"workflow_id": wf_id})

    if len(docs):
        print("Found docs!")

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
    wf_obj_file = os.path.join(rep_dir, f"wf_obj_{wf_id}")
    with open(wf_obj_file, "w") as json_file:
        json.dump(wfobj.to_dict(), json_file, indent=2)

    print(f"Saved files {dump_file} and {wf_obj_file}.")
