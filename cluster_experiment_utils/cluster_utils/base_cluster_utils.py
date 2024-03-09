import json
import os
import subprocess
from abc import ABCMeta
from datetime import datetime, timedelta
from uuid import uuid4
from typing import Dict, Union

from omegaconf import OmegaConf
import psij

from cluster_experiment_utils.utils import get_resource_manager_type, get_runner_type


class ResourceManager:
    LSF = "lsf"
    SLURM = "slurm"
    PBS = "pbs"


class Runner:
    JSRUN = "jsrun"
    SRUN = "srun"


class BaseClusterUtils(object, metaclass=ABCMeta):
    _subclasses = {}

    @classmethod
    def register_subclass(cls, subclass_name, subclass):
        cls._subclasses[subclass_name] = subclass

    _instance = None
    RESOURCE_MANAGER = get_resource_manager_type()

    @staticmethod
    def get_instance():
        if BaseClusterUtils._instance is not None:
            return BaseClusterUtils._instance
        else:
            if BaseClusterUtils.RESOURCE_MANAGER == ResourceManager.SLURM:
                BaseClusterUtils._instance = BaseClusterUtils._subclasses.get(
                    ResourceManager.SLURM
                )()
            elif BaseClusterUtils.RESOURCE_MANAGER == ResourceManager.LSF:
                BaseClusterUtils._instance = BaseClusterUtils._subclasses.get(
                    ResourceManager.LSF
                )()
        return BaseClusterUtils._instance

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

    def kill_all_running_job_steps():
        raise NotImplementedError()

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
        raise NotImplementedError()
        # return self._generic_job_submission(
        #     cmd=cmd,
        #     stdout=stdout,
        #     stderr=stderr,
        #     node_count=node_count,
        #     process_count=process_count,
        #     processes_per_node=processes_per_node,
        #     cpu_cores_per_process=cpu_cores_per_process,
        #     gpu_cores_per_process=gpu_cores_per_process,
        #     job_type="runner",
        # )

    def submit_batch_job(
        self,
        cmd,
        proj_id,
        queue_name=None,
        job_name=None,
        wall_time: str = None,  # format: HH:MM
        stdout=None,
        stderr=None,
        node_count=None,
        process_count=None,
        processes_per_node=None,
        custom_attributes=None,
        cpu_cores_per_process=None,
        gpu_cores_per_process=None,
    ):
        self._generic_job_submission(
            cmd=cmd,
            proj_id=proj_id,
            queue_name=queue_name,
            job_name=job_name,
            wall_time=wall_time,
            stdout=stdout,
            stderr=stderr,
            node_count=node_count,
            process_count=process_count,
            custom_attributes=custom_attributes,
            processes_per_node=processes_per_node,
            cpu_cores_per_process=cpu_cores_per_process,
            gpu_cores_per_process=gpu_cores_per_process,
            job_type="batch",
        )

    def _generic_job_submission(
        self,
        cmd,
        proj_id=None,
        queue_name=None,
        job_name=None,
        wall_time: str = None,  # format: HH:MM
        stdout=None,
        stderr=None,
        node_count=None,
        process_count=None,
        processes_per_node=None,
        custom_attributes=None,
        cpu_cores_per_process=None,
        gpu_cores_per_process=None,
        job_type="batch",  # or "runner"
    ) -> Union[psij.Job, subprocess.Popen]:
        job = BaseClusterUtils._build_job(
            cmd=cmd,
            proj_id=proj_id,
            queue_name=queue_name,
            job_name=job_name,
            wall_time=wall_time,
            stdout=stdout,
            stderr=stderr,
            node_count=node_count,
            process_count=process_count,
            processes_per_node=processes_per_node,
            custom_attributes=custom_attributes,
            cpu_cores_per_process=cpu_cores_per_process,
            gpu_cores_per_process=gpu_cores_per_process,
        )
        if job_type == "batch":
            jex = psij.JobExecutor.get_instance(BaseClusterUtils.RESOURCE_MANAGER)
            print("Going to submit job")
            jex.submit(job)
            print("Submitted!")
            return job
        else:
            raise Exception("Unexpected")
        # Unfortunately I couldn't make psij work for job steps.
        # elif job_type == "runner":
        #     return BaseClusterUtils._launch_job(job)

    @staticmethod
    def _launch_job(job: psij.Job) -> subprocess.Popen:
        runner = get_runner_type(BaseClusterUtils.RESOURCE_MANAGER)
        launcher = psij.Launcher.get_instance(runner)
        command = launcher.get_launch_command(job)
        print(f"Going to run the command: {command}")
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return process

    @staticmethod
    def _build_job(
        cmd,
        proj_id=None,
        queue_name=None,
        job_name=None,
        wall_time: str = None,  # format: HH:MM
        stdout=None,
        stderr=None,
        node_count=None,
        process_count=None,
        processes_per_node=None,
        custom_attributes=None,
        cpu_cores_per_process=None,
        gpu_cores_per_process=None,
    ) -> psij.Job:
        # shell_script_file = os.path.join(str(uuid4())+".sh")
        # print(f"Written the following command into the file {shell_script_file}\n{cmd}")
        # with open(shell_script_file, "w+") as f:
        #     f.write(cmd)

        cmd_split = cmd.split()

        job = psij.Job()
        spec = psij.JobSpec()
        spec.executable = cmd_split[0]
        spec.arguments = cmd_split[1:]
        print(spec.executable, spec.arguments)
        # cmds = cmd.split()
        # spec.executable = " ".join(cmds)
        # print(spec.executable)
        # spec.arguments = cmds[1:]

        spec.name = job_name
        spec.attributes.project_name = proj_id
        spec.attributes.queue_name = queue_name
        spec.attributes.custom_attributes = custom_attributes
        if custom_attributes:
            print(custom_attributes)
        if wall_time:
            spec.attributes.duration = BaseClusterUtils._parse_walltime_string(
                wall_time
            )

        spec.stderr_path = stderr
        spec.stdout_path = stdout

        resource = psij.ResourceSpecV1()
        resource.node_count = node_count
        resource.process_count = process_count
        resource.processes_per_node = processes_per_node
        resource.cpu_cores_per_process = cpu_cores_per_process
        resource.gpu_cores_per_process = gpu_cores_per_process

        spec.resources = resource
        job.spec = spec

        return job

    def generate_job_output(
        self,
        conf_data,
        job_hosts,
        job_dir,
        my_job_id,
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
            "job_id": self.get_this_job_id(),
            "job_hosts": job_hosts,
            "varying_param_key": varying_param_key,
            "rep_no": rep_no,
            "with_flowcept": with_flowcept,
            "python_env": python_env,
            "run_end_time": datetime.utcnow().strftime("%Y-%m-%d %H-%M-%S.%f")[:-3],
            "total_time": t1 - t0,
            "client_time": t_c_f - t_c_i,
        }
        if wf_result is not None:
            out_job["wf_result"] = wf_result
        out_job["exp_settings"] = OmegaConf.to_container(conf_data)
        if flowcept_settings is not None:
            out_job["flowcept_settings"] = OmegaConf.to_container(flowcept_settings)
        print(json.dumps(out_job, indent=2))
        with open(f"{rep_dir}/out_job.json", "w") as f:
            f.write(json.dumps(out_job, indent=2) + "\n")
        with open(f"{proj_dir}/results.jsonl", "a+") as f:
            f.write(json.dumps(out_job) + "\n")
        return out_job

    @staticmethod
    def _parse_walltime_string(walltime_str):
        try:
            hours, minutes = map(int, walltime_str.split(":"))
            walltime = timedelta(hours=hours, minutes=minutes)
            return walltime
        except ValueError:
            raise ValueError("Invalid walltime string format")
