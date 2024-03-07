import os


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
        print("Killing LSF job")
        run_cmd_check_output(f"scancel {job_id}")
        print("Kill command submitted!")

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
