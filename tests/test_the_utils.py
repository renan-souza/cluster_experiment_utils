import os.path
import unittest
from pathlib import Path
import random

from omegaconf import OmegaConf

from experiment_utils.flowcept_utils import update_flowcept_settings
from experiment_utils.utils import generate_configs


class TestTheUtils(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTheUtils, self).__init__(*args, **kwargs)

    def test(self):
        exp_param_settings = {
            "param_name1": {
                "init": 1,
                "end": 3,
                "step": 1,
            },
            "param_name2": {"init": [100, 200], "end": [500, 600], "step": 100},
            "param_name4": {"init": 0.1, "end": 0.9, "step": 0.1},
            "param_name3": ["A", "B", "C"],
            "param_name5": [1e-1, 1e-2, 1e-3],
        }
        print(generate_configs(exp_param_settings))

    def test_update_flowcept(self):
        flowcept_yaml_path = "../conf/sample_flowcept_settings.yaml"
        exp_conf_yaml_path = "../conf/exp_params.yaml"
        flowcept_settings = OmegaConf.load(Path(flowcept_yaml_path))
        exp_conf = OmegaConf.load(Path(exp_conf_yaml_path))
        rep_dir = os.path.expanduser(f"~/.flowcept/tests/rep0")
        os.makedirs(rep_dir, exist_ok=True)
        update_flowcept_settings(
            exp_conf=exp_conf,
            flowcept_settings=flowcept_settings,
            db_host="dblocalhost",
            should_start_mongo=True,
            repetition_dir=rep_dir,
            varying_param_key="small_test",
            job_id=1234,
        )
