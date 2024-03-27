import sys
import argparse
import os
import json
import pprint
from time import sleep 
import pandas as pd
from uuid import uuid4

from cluster_experiment_utils.utils import generate_configs


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


def dask_workflow(
    client, llm_run_meta_params, rep_dir, input_data_dir, model_train_func, workflow_id
):
    wf_id = workflow_id  # f"wf_{uuid4()}"
    print(f"Workflow_Id={wf_id}")
    with open(os.path.join(rep_dir, "parent_wf_" + wf_id + ".txt"), "w+") as f:
        f.write(wf_id)

    with open(os.path.join(args.rep_dir, "dask_client_info.json"), "w") as json_file:
        json.dump(dict(client.scheduler_info()), json_file, indent=2)

    configs = generate_configs(llm_run_meta_params)
    with open(os.path.join(args.rep_dir, "llm_configs.txt"), "w") as file:
        pretty_printer = pprint.PrettyPrinter(stream=file)
        pretty_printer.pprint(configs)

    outputs = []
    for conf in configs:
        print(conf)
        conf.update(
            {
                "workflow_id": wf_id,
                "input_data_dir": input_data_dir,
                "rep_dir": rep_dir,
            }
        )
        outputs.append(client.submit(model_train_func, **conf))

    results = []
    times = []
    for o in outputs:
        r = o.result()
        print("Done one model")
        r.pop("model", None)  # removing unserializable
        results.append(r)

    df = pd.DataFrame(results)
    mean_times = dict(df.filter(like="time", axis=1).mean())
    with open(os.path.join(rep_dir, "mean_times.json"), "w") as outfile:
        json.dump(mean_times, outfile)

    with open(os.path.join(rep_dir, "workflow_result.json"), "w") as outfile:
        json.dump(
            {
                "workflow_id": wf_id,
                "results": results,
                "dask_info": client.scheduler_info(),
            },
            outfile,
        )

    
    print("I'm Dask client. I'm going to close Dask gracefully! Gonna wait 60s before closing.")
    sleep(30)
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
    required.add_argument("--workflow-id", metavar="D", required=True, help="Wf Id")
    required.add_argument(
        "--workflow-params",
        metavar="D",
        required=True,
        help="Workflow Parameters as a sringfied dictionary",
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

    workflow_params_str = args.workflow_params
    llm_run_meta_params = json.loads(workflow_params_str)
    input_data_dir = llm_run_meta_params.pop("input_data_dir")
    print(input_data_dir)
    with open(os.path.join(args.rep_dir, "llm_meta_params.json"), "w") as json_file:
        json.dump(
            llm_run_meta_params, json_file, indent=2
        )  # indent for pretty formatting (optional)

    if args.with_flowcept:
        from executors.dask_workflows.llm.llm_trainer import model_train

        model_train_func = model_train
    else:
        from executors.dask_workflows.llm.llm_trainer_no_flowcept import model_train

        model_train_func = model_train

    dask_workflow(
        client,
        llm_run_meta_params,
        args.rep_dir,
        input_data_dir,
        model_train_func,
        args.workflow_id,
    )
