from abc import ABCMeta
from typing import Dict


class BaseClusterUtils(object, metaclass=ABCMeta):
    def __init__(self):
        pass

    def kill_job(self, job_id):
        raise NotImplementedError()

    def get_this_job_id(self):
        raise NotImplementedError()

    def get_job_hosts(self):
        raise NotImplementedError()

    def get_resource_usage_info(self, job_dir) -> Dict:
        raise NotImplementedError()
