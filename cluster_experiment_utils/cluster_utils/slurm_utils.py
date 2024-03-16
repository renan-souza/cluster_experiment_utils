import os
import subprocess
from cluster_experiment_utils.cluster_utils.base_cluster_utils import (
    BaseClusterUtils,
)
from cluster_experiment_utils.utils import run_cmd_check_output, printed_sleep


class SlurmUtils(BaseClusterUtils):
    def get_this_job_id(self):
        return os.getenv("SLURM_JOB_ID")

    def kill_job(self, job_id=None):
        if job_id is None:
            job_id = self.get_this_job_id()
        print(f"Killing job {job_id}")
        run_cmd_check_output(f"scancel {job_id}")
        print("Kill command submitted!")
        printed_sleep(2)

    def kill_all_running_job_steps(self):
        this_job = self.get_this_job_id()
        try:
            # Run the sacct command and capture the output
            result = subprocess.run(
                ["sacct", "-j", str(this_job)],
                capture_output=True,
                text=True,
                check=True,
            )
            # Split the output into lines and iterate through them
            for line in result.stdout.strip().split("\n"):
                # Split each line into columns
                columns = line.split()
                # Check if the line corresponds to a RUNNING job
                if len(columns):
                    step_job_id = columns[0]
                    status = columns[4]
                    if "." in step_job_id and status == "RUNNING":
                        split_val = step_job_id.split(".")
                        if split_val[1].isdigit():
                            self.kill_job(step_job_id)

        except subprocess.CalledProcessError as e:
            print(f"Error running sacct: {e}")

    def run_job(
        self,
        cmd,
        stdout=None,
        stderr=None,
        node_count=None,
        process_count=None,
        processes_per_node=None,
        cpu_cores_per_process=None,
        gpus_per_job=None,
    ):
        srun_command = ["srun", "--exclusive"]

        if node_count is not None:
            srun_command.extend(["--nodes", str(node_count)])

        if process_count is not None:
            srun_command.extend(["--ntasks", str(process_count)])

        if processes_per_node is not None:
            srun_command.extend(["--ntasks-per-node", str(processes_per_node)])

        if cpu_cores_per_process is not None:
            srun_command.extend(["--cpus-per-task", str(cpu_cores_per_process)])

        if gpus_per_job is not None:
            srun_command.extend(["--gpus", str(gpus_per_job)])

        if stdout is not None:
            srun_command.extend(["--output", stdout])

        if stderr is not None:
            srun_command.extend(["--error", stderr])

        cmd = cmd.strip()
        srun_command_str = " ".join(srun_command)
        srun_command_str += " /bin/bash -c '" + cmd + "'"
        print(srun_command_str)
        process = subprocess.Popen(srun_command_str, shell=True)

        return process

    def get_job_hosts(self):
        """
        Gets a list of job hosts
        :return:
        """
        output = run_cmd_check_output("scontrol show hostnames $SLURM_JOB_NODELIST")
        hosts = output.split()
        assert len(hosts)
        return hosts

    def __init__(self):
        super().__init__()


BaseClusterUtils.register_subclass("slurm", SlurmUtils)
