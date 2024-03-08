import os
import subprocess

from cluster_experiment_utils.cluster_utils.base_cluster_utils import (
    BaseClusterUtils,
)
from cluster_experiment_utils.utils import run_cmd_check_output


class SlurmUtils(BaseClusterUtils):
    def get_this_job_id(self):
        return os.getenv("SLURM_JOB_ID")

    def kill_job(self, job_id=None):
        if job_id is None:
            job_id = self.get_this_job_id()
        print(f"Killing job {job_id}")
        run_cmd_check_output(f"scancel {job_id}")
        print("Kill command submitted!")


    def kill_all_running_job_steps(self):
        this_job = self.get_this_job_id()        
        try:
            # Run the sacct command and capture the output
            result = subprocess.run(['sacct', '-j', str(this_job)], capture_output=True, text=True, check=True)
            # Split the output into lines and iterate through them
            for line in result.stdout.strip().split('\n'):
                # Split each line into columns
                columns = line.split()
    
                # Check if the line corresponds to a RUNNING job
                if len(columns) > 5 and columns[5] == 'RUNNING':
                    step_job_id = columns[0]
                    print(step_job_id)
                    if "." in step_job_id:
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
        srun_command = ['srun', '--exclusive']

        if node_count is not None:
            srun_command.extend(['--nodes', str(node_count)])
    
        if process_count is not None:
            srun_command.extend(['--ntasks', str(process_count)])
    
        if processes_per_node is not None:
            srun_command.extend(['--ntasks-per-node', str(processes_per_node)])
    
        if cpu_cores_per_process is not None:
            srun_command.extend(['--cpus-per-task', str(cpu_cores_per_process)])
    
        if gpus_per_job is not None:
            srun_command.extend(['--gpus', str(gpus_per_job)])
    
        if stdout is not None:
            srun_command.extend(['--output', stdout])
    
        if stderr is not None:
            srun_command.extend(['--error', stderr])

        cmd = cmd.strip()
        srun_command_str = " ".join(srun_command)
        srun_command_str += " /bin/bash -c '" + cmd + "'"
        #srun_command.extend(cmd.split())
        print(srun_command_str)
        process = subprocess.Popen(srun_command_str, shell=True)
        
        return process

    def get_job_hosts(self):
        """
        Gets a mapping of job hosts and number of available cores per host
        :return:
        """
        # TODO : revisit this
        hosts = os.getenv("SLURM_JOB_NODELIST")
        if hosts is not None:
            lsb_hosts = hosts.split()
        else:
            host_file = os.getenv("SLURM_JOB_NODEFILE")
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
        #
        # def get_resource_usage_info(self, job_dir) -> Dict:
        #     return {}
        #     lsf_job_id = os.getenv("LSB_JOBID")
        #     output = run_cmd_check_output(f"bjobs -l {lsf_job_id}")
        #     with open(f"{job_dir}/resources_usage.txt", "a+") as f:
        #         f.write(output)
        #
        #     float_point_regex = "[+-]?([0-9]*[.])?[0-9]+"
        #     match = re.search(
        #         r"CPU time used is (" + float_point_regex + ") seconds", output
        #     )
        #     cpu_time = -1
        #     if match:
        #         cpu_time = float(match.group(1))
        #
        #     match = re.search(r"MAX MEM: (\d+) Mbytes", output)
        #     max_mem = -1
        #     if match:
        #         max_mem = int(match.group(1))
        #
        #     match = re.search(r"AVG MEM: (\d+) Mbytes", output)
        #     avg_mem = -1
        #     if match:
        #         avg_mem = int(match.group(1))
        #
        #     match = re.search(r"Submitted from host <([a-zA-Z0-9]+)>", output)
        #     from_host = None
        #     if match:
        #         from_host = match.group(1)

        # TODO: implement
        return {
            "lsf_cpu_time": 0,
            "lsf_max_mem_mb": 0,
            "lsf_avg_mem_mb": 0,
            "from_host": 0,
        }

    def __init__(self):
        super().__init__()


BaseClusterUtils.register_subclass("slurm", SlurmUtils)
