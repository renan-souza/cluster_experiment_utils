import json
from abc import ABCMeta
from datetime import datetime
from typing import Dict

from build.lib.cluster_experiment_utils.cluster_utils import base_cluster_utils
from cluster_experiment_utils import cluster_utils


class BaseClusterUtils(object):
    def __init__(self):
        pass

    def kill_job(self, job_id):
        raise NotImplementedError()

    def get_this_job_id(self):
        raise NotImplementedError()

    def kill_this_job(self):
        _id = self.get_this_job_id()
        self.kill_job(_id)

    def get_job_hosts(self):
        """
        Gets a mapping of job hosts and number of available cores per host
        :return:
        """
        raise NotImplementedError()

    def get_resource_usage_info(self, job_dir) -> Dict:
        raise NotImplementedError()

    def generate_job_output(
        self,
        cluster_utils,
        conf_data,
        host_allocs,
        host_counts,
        job_dir,
        my_job_id,
        scheduler_job_id,
        proj_dir,
        python_env,
        rep_dir,
        rep_no,
        t0,
        t1,
        t_c_f,
        t_c_i,
        varying_param_key,
        wf_result,
        with_flowcept,
        flowcept_settings,
    ):
        out_job = {
            "my_job_id": my_job_id,
            "job_dir": job_dir,
            "rep_dir": rep_dir,
            "lsf_job_id": scheduler_job_id,
            "lsb_hosts": host_counts,
            "varying_param_key": varying_param_key,
            "rep_no": rep_no,
            "with_flowcept": with_flowcept,
            "python_env": python_env,
            "run_end_time": datetime.utcnow().strftime("%Y-%m-%d %H-%M-%S.%f")[:-3],
            "total_time": t1 - t0,
            "client_time": t_c_f - t_c_i,
        }

        resource_usage = None
        try:
            resource_usage = cluster_utils.get_resource_usage_info(rep_dir)
        except Exception as e:
            print("Could not retrieve resource usage")
            print(e)

        if wf_result is not None:
            out_job["wf_result"] = wf_result
        out_job["exp_settings"] = conf_data
        if flowcept_settings is not None:
            out_job["flowcept_settings"] = flowcept_settings
        out_job.update(host_allocs)
        if resource_usage is not None:
            out_job.update(resource_usage)
        print(json.dumps(out_job, indent=2))
        with open(f"{rep_dir}/out_job.json", "w") as f:
            f.write(json.dumps(out_job, indent=2) + "\n")
        with open(f"{proj_dir}/results.jsonl", "a+") as f:
            f.write(json.dumps(out_job) + "\n")
        return out_job
