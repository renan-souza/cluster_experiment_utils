import os
import re
from typing import Dict
import subprocess

from cluster_experiment_utils.cluster_utils.base_cluster_utils import BaseClusterUtils
from cluster_experiment_utils.utils import run_cmd_check_output


class LsfUtils(BaseClusterUtils):
    def get_this_job_id(self):
        return os.getenv("LSB_JOBID")

    def kill_job(self, job_id=None):
        if job_id is None:
            job_id = self.get_this_job_id()
        print("Killing LSF job")
        run_cmd_check_output(f"bkill {job_id}")
        print("Kill command submitted!")

    def kill_all_running_job_steps(self):
        print("Killing all jsrun jobs")
        run_cmd_check_output(f"jskill all")

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
        jsrun_command = [
            "jsrun",
            "--nrs",
            str(node_count) if node_count else "1",
            "--tasks_per_rs",
            str(process_count) if process_count else "1",
            "--tasks_per_host",
            str(processes_per_node) if processes_per_node else "1",
            "--cpu_per_rs",
            str(cpu_cores_per_process) if cpu_cores_per_process else "1",
            "--gpu_per_rs",
            str(gpu_cores_per_process) if gpu_cores_per_process else "0",
            "--stdout",
            stdout,
            "--stderr",
            stderr,
        ]

        jsrun_command += cmd
        print(jsrun_command)
        process = subprocess.Popen(jsrun_command)
        return process

    def get_job_hosts(self):
        hosts = os.getenv("LSB_HOSTS")
        if hosts is not None:
            lsb_hosts = hosts.split()
        else:
            host_file = os.getenv("LSB_DJOB_HOSTFILE")
            with open(host_file) as f:
                lsb_hosts = f.read().split("\n")

        host_counts = dict()
        for h in lsb_hosts:
            if not h:
                continue
            if h not in host_counts:
                host_counts[h] = 1
            else:
                host_counts[h] += 1
        print(host_counts)
        return host_counts

    def get_resource_usage_info(self, job_dir) -> Dict:
        lsf_job_id = os.getenv("LSB_JOBID")
        output = run_cmd_check_output(f"bjobs -l {lsf_job_id}")
        with open(f"{job_dir}/resources_usage.txt", "a+") as f:
            f.write(output)

        float_point_regex = "[+-]?([0-9]*[.])?[0-9]+"
        match = re.search(
            r"CPU time used is (" + float_point_regex + ") seconds", output
        )
        cpu_time = -1
        if match:
            cpu_time = float(match.group(1))

        match = re.search(r"MAX MEM: (\d+) Mbytes", output)
        max_mem = -1
        if match:
            max_mem = int(match.group(1))

        match = re.search(r"AVG MEM: (\d+) Mbytes", output)
        avg_mem = -1
        if match:
            avg_mem = int(match.group(1))

        match = re.search(r"Submitted from host <([a-zA-Z0-9]+)>", output)
        from_host = None
        if match:
            from_host = match.group(1)

        return {
            "lsf_cpu_time": cpu_time,
            "lsf_max_mem_mb": max_mem,
            "lsf_avg_mem_mb": avg_mem,
            "from_host": from_host,
        }

    def __init__(self):
        super().__init__()


BaseClusterUtils.register_subclass("lsf", LsfUtils)
