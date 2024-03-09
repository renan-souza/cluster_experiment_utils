import sys
import argparse
import os
import json
from pathlib import Path
from uuid import uuid4

from omegaconf import OmegaConf

from cluster_experiment_utils.utils import generate_configs

from executors.dask_workflows.llm.llm_trainer import get_wiki_text, model_train


def init_dask(scheduler_file, with_flowcept=False):
    from time import sleep
    from dask.distributed import Client

    client = Client(scheduler_file=scheduler_file)
    consumer = None
    if with_flowcept:
        from flowcept import FlowceptDaskWorkerAdapter

        client.register_worker_plugin(FlowceptDaskWorkerAdapter())
        sleep(2)

    return client, consumer


def dask_workflow(client, llm_run_meta_params, rep_dir):
    ntokens, train_data, val_data, test_data = get_wiki_text()

    wf_id = f"wf_{uuid4()}"
    print(f"Workflow_Id={wf_id}")
    configs = generate_configs(llm_run_meta_params)
    outputs = []
    for conf in configs:
        conf.update(
            {
                "ntokens": ntokens,
                "train_data": train_data,
                "val_data": val_data,
                "test_data": test_data,
                "workflow_id": wf_id,
            }
        )
        outputs.append(client.submit(model_train, **conf))

    results = []
    for o in outputs:
        r = o.result()
        r.pop("model")  # removing unserializable
        results.append(r)

    with open(os.path.join(rep_dir, "workflow_result.json"), "w") as outfile:
        json.dump({"workflow_id": wf_id, "results": results}, outfile)
    print("I'm Dask client. I'm going to close Dask gracefully!")
    client.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Submit Dask workflow.")
    optional = parser._action_groups.pop()

    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--scheduler-file", metavar="S", required=True, help="Dask's scheduler file"
    )
    required.add_argument(
        "--rep-dir", metavar="D", required=True, help="Job's repetition directory"
    )
    required.add_argument(
        "--exp-conf", metavar="D", required=True, help="Experiment Conf File"
    )
    required.add_argument(
        "--varying-param-key", metavar="D", required=True, help="Varying Param Key"
    )

    optional.add_argument(
        "--with-flowcept", action="store_true", help="Enable Flowcept interceptions"
    )

    parser._action_groups.append(optional)

    if len(sys.argv) == 1:
        parser.print_help()

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print("Arguments in llm_workflow.py=", args)
    client, consumer = init_dask(args.scheduler_file, args.with_flowcept)
    print("client", client, consumer)

    exp_conf_data = OmegaConf.load(Path(args.exp_conf))
    varying_param_key = args.varying_param_key
    llm_run_meta_params = exp_conf_data.varying_params.varying_param_key.workflow_params
    print(llm_run_meta_params)
    dask_workflow(client, llm_run_meta_params, args.rep_dir)
    # if consumer is not None:
    #     consumer.stop()
