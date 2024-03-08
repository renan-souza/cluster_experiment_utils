import sys
import argparse
import os


def dummy_func1(x, workflow_id=None):
    return x * 2


def dummy_func2(y, workflow_id=None):
    return y + y


def calculate_batch_and_epochs(z, w, workflow_id=None):
    result = {"batch_size": int(z + w + 16), "epochs": max(int(z / w) + 1, 2)}
    # Silly thing just to check GPU info
    import torch

    if torch.cuda.is_available():
        gpus = {}
        for i in range(torch.cuda.device_count()):
            gpus[i] = torch.cuda.get_device_properties(i).name
        result["gpus"] = gpus
    return result


def init_dask(rep_dir, scheduler_file, with_flowcept=False):
    from time import sleep
    from dask.distributed import Client

    client = Client(scheduler_file=scheduler_file)
    consumer = None
    if with_flowcept:
        from flowcept import FlowceptDaskWorkerAdapter

        client.register_worker_plugin(FlowceptDaskWorkerAdapter())
        sleep(2)

        # from flowcept import FlowceptConsumerAPI
        # consumer = FlowceptConsumerAPI()
        # consumer.start()

    return client, consumer


def dask_workflow(client, rep_dir):
    import json
    import numpy as np
    from uuid import uuid4

    i1 = np.random.random()
    wf_id = f"wf_{uuid4()}"
    print(f"Workflow_Id={wf_id}")
    o1 = client.submit(dummy_func1, i1, workflow_id=wf_id)
    o2 = client.submit(dummy_func2, o1, workflow_id=wf_id)
    o3 = client.submit(calculate_batch_and_epochs, o1, o2, workflow_id=wf_id)
    print(f"Task3_id={o3.key}")
    print(f"Result={o3.result()}")
    with open(os.path.join(rep_dir, "workflow_result.json"), "w") as outfile:
        json.dump(
            {"workflow_id": wf_id, "task_id": o3.key, "result": o3.result()}, outfile
        )
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
    print("Arguments in simple_dask_workflow.py=", args)
    client, consumer = init_dask(args.rep_dir, args.scheduler_file, args.with_flowcept)
    print("client", client, consumer)
    dask_workflow(client, args.rep_dir)
    # if consumer is not None:
    #     consumer.stop()
