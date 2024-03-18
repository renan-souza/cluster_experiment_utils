import argparse
from omegaconf import OmegaConf
import os
import socket
import pymongo

from cluster_experiment_utils.flowcept_utils import start_mongo


def main():
    parser = argparse.ArgumentParser(
        description="Process YAML configuration file and update with hostname."
    )
    parser.add_argument(
        "--exp_conf",
        required=True,
        help="Path to the exp_conf YAML configuration file.",
    )
    parser.add_argument(
        "--flowcept_conf",
        required=True,
        help="Path to the flowcept_conf YAML configuration file.",
    )
    args = parser.parse_args()

    # Load YAML configuration using OmegaConf for flowcept_conf
    flowcept_conf = OmegaConf.load(args.flowcept_conf)

    host = flowcept_conf.mongodb.host
    port = flowcept_conf.mongodb.port
    client = pymongo.MongoClient(host=host, port=port, connectTimeoutMS=1000)

    # Select the database and collection
    db = client[flowcept_conf.mongodb.db]
    collection = db[flowcept_conf.mongodb.collection]

    # Perform a simple query to check the connection
    print(
        f"Checking if MongoDB on host {host} is runnig. If it's down, it will wait until timeout."
    )
    try:
        result = collection.find_one()
        print(f"MongoDB on host {host} is already up!")
        return
    except Exception as e:
        print(e)

    # Execute the start_mongo_cmd in exp_conf
    exp_conf = OmegaConf.load(args.exp_conf)

    mongo_start_cmd = exp_conf.static_params.mongo_start_command
    mongo_dir = os.path.join(exp_conf.static_params.proj_dir, "mongo")
    start_mongo(host, port, mongo_start_cmd, mongo_dir)

    # Get current hostname
    hostname = socket.gethostname()

    # Update YAML configuration in flowcept_conf with the current hostname
    OmegaConf.update(flowcept_conf, "mongodb.host", hostname)

    # Save the updated configuration back to the flowcept_conf file
    OmegaConf.save(flowcept_conf, args.flowcept_conf)

    print(f"Updated flowcept_conf with current hostname: {hostname}")


if __name__ == "__main__":
    main()
