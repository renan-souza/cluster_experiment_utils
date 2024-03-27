import json
import os
import redis
import time
from pathlib import Path
from typing import Any, Dict
import getpass
import pymongo


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
    job_hosts,
    should_start_mongo,
    repetition_dir,
    varying_param_key,
    job_id,
    nnodes,
    n_workers=None,
):
    log_path = os.path.join(repetition_dir, "flowcept.log")
    new_settings = OmegaConf.create(flowcept_settings)
    instances = None
    if flowcept_settings.main_redis.get("instances", None) is not None:
        redis_policy = exp_conf.static_params.redis_policy
        if redis_policy == "one_per_worker":
            instances = []
            n_instances_per_node = int(n_workers / nnodes)
            init_port = 6379
            for i in range(nnodes):
                job_host = job_hosts[i]
                for j in range(0, n_instances_per_node):
                    _port = init_port + j
                    instances.append(f"{job_host}:{_port}")
        elif redis_policy == "one":
            instances = [f"{job_hosts[0]}:6379"]
        else:
            raise NotImplementedError()

    flowcept_file_log_level = exp_conf.static_params.flowcept_file_log_level
    flowcept_stdout_log_level = exp_conf.static_params.flowcept_stdout_log_level
    db_host = job_hosts[0]
    new_settings = omegaconf_simple_variable_mapping(
        new_settings,
        variable_mapping={
            "db_host": db_host,
            "redis_instances": instances,
            "flowcept_file_log_level": flowcept_file_log_level,
            "flowcept_stdout_log_level": flowcept_stdout_log_level,
            "job_id": job_id,
            "log_path": log_path,
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


def kill_dbs(flowcept_settings, should_start_mongo):
    redis_instances = flowcept_settings.main_redis.get("instances", None)
    if redis_instances is None:
        redis_host = flowcept_settings.main_redis.host
        redis_port = flowcept_settings.main_redis.port
        print("Killing redis...")
        run_cmd(f"ssh {redis_host} pkill -9 -f redis-server &")
    else:
        for redis_instance in redis_instances:
            split = redis_instance.split(":")
            redis_host = split[0]
            run_cmd(f"ssh {redis_host} pkill -9 -f redis-server &")
    if should_start_mongo:
        mongo_host = flowcept_settings.mongodb.host
        print("Killing mongo...")
        run_cmd(f"ssh {mongo_host} pkill mongod &")

    printed_sleep(20)


def start_mongo(mongo_host, mongo_port, mongo_start_cmd, rep_dir):
    print("Starting MongoDB...")
    mongo_data_dir = os.path.join(rep_dir, "mongo_data")
    mongo_data_dir_db = os.path.join(mongo_data_dir, "db")
    os.makedirs(mongo_data_dir_db, exist_ok=True)
    mongo_log = os.path.join(mongo_data_dir, "mongo.log")
    open(mongo_log, "w").close()

    variable_mapping = {"MONGO_DATA": mongo_data_dir_db, "MONGO_LOG": mongo_log}
    mongo_start_cmd = replace_var_mapping_in_str(mongo_start_cmd, variable_mapping)

    if "local" in mongo_host:
        run_cmd(f"{mongo_start_cmd} &")
    else:
        run_cmd(f"ssh {mongo_host} {mongo_start_cmd} & ")

    printed_sleep(10)

    start_time = time.time()
    trial = 0
    max_time = 60
    max_trials = 30

    while True:
        trial += 1
        if max_time and time.time() - start_time > max_time:
            print("Timeout reached. Exiting...")
            break

        try:
            client = pymongo.MongoClient(
                host=mongo_host, port=mongo_port, connectTimeoutMS=1000
            )
            client.server_info()
            print(f"MongoDB server {mongo_host}:{mongo_port} is up and running.")
            print("Creating index...")

            from flowcept.commons.daos.document_db_dao import DocumentDBDao

            DocumentDBDao(create_index=True)  # this will force index creation

            print("Created mongo index!")
            break
        except pymongo.errors.ConnectionFailure as e:
            print(
                f"Failed to connect {mongo_host}:{mongo_port} (Trial {trial}/{max_trials}): {e}"
            )
            if max_trials and trial == max_trials:
                print("Max trials reached. Exiting...")
                break
            time.sleep(1)


def start_redis(flowcept_settings, exp_conf, rep_dir):
    base_redis_start_command = exp_conf.static_params.redis_start_command
    redis_instances = flowcept_settings.main_redis.get("instances", None)
    if redis_instances is None:
        redis_host = flowcept_settings.main_redis.host
        redis_port = flowcept_settings.main_redis.port
        start_redis_instance(redis_start_command, redis_host, redis_port)
    else:
        redis_log_dir = os.path.join(rep_dir, "redis_logs")
        os.makedirs(redis_log_dir, exist_ok=True)
        for redis_instance in redis_instances:
            redis_log_name = redis_instance.replace(":", "_") + ".log"
            redis_log_path = os.path.join(redis_log_dir, redis_log_name)
            split = redis_instance.split(":")
            redis_host = split[0]
            redis_port = split[1]
            redis_start_command = (
                base_redis_start_command
                + f" --logfile {redis_log_path} --port {redis_port}"
            )
            start_redis_instance(redis_start_command, redis_host, redis_port)


def start_redis_instance(redis_start_cmd, db_host, db_port):
    print("Starting Redis")
    run_cmd(f"ssh {db_host} {redis_start_cmd} &")
    printed_sleep(2)

    start_time = time.time()
    trials = 0
    success = False
    max_trials = 10
    check_interval = 10
    max_duration = 120

    while trials < max_trials and time.time() - start_time < max_duration:
        try:
            r = redis.StrictRedis(host=db_host, port=db_port, decode_responses=True)
            r.ping()
            success = True
            break
        except redis.ConnectionError as e:
            print(
                f"Trial {trials + 1}: Redis connection {db_host}:{db_port} failed. Retrying in {check_interval} seconds."
            )
            # run_cmd(f"ssh {db_host} pkill -9 -f redis-server &")
            # printed_sleep(check_interval)
            # run_cmd(f"ssh {db_host} {redis_start_cmd} &")
            printed_sleep(check_interval)
            trials += 1

    if success:
        print(f"Redis is alive on {db_host}:{db_port}!")
    else:
        raise Exception(
            f"Unable to establish a connection to Redis on {db_host} after {max_trials} trials."
        )


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
