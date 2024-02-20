import subprocess
from time import sleep
import itertools


def printed_sleep(secs):
    print(f"Going to wait for {secs} secs...")
    sleep(secs)


def run_cmd_check_output(cmd: str):
    print(f"Running command:\n{cmd}")
    cmd_output = subprocess.check_output(cmd, shell=True)
    cmd_output_str = cmd_output.decode().strip()
    return cmd_output_str


def run_cmd(cmd: str):
    print(f"Running command:\n{cmd}")
    subprocess.Popen(cmd, shell=True)


def _interpolate_values(start, end, step):
    return [start + i * step for i in range((end - start) // step + 1)]


def generate_configs(params):
    param_names = list(params.keys())
    param_values = []

    for param_name in param_names:
        param_data = params[param_name]

        if isinstance(param_data, dict):
            init_value = param_data["init"]
            end_value = param_data["end"]
            step_value = param_data.get("step", 1)

            if isinstance(init_value, (int, float)):
                param_values.append(
                    [
                        round(val / 10, 1)
                        for val in range(
                            int(init_value * 10),
                            int((end_value + step_value) * 10),
                            int(step_value * 10),
                        )
                    ]
                )
            elif isinstance(init_value, list) and all(
                isinstance(v, (int, float)) for v in init_value
            ):
                interpolated_values = _interpolate_values(
                    init_value[0], end_value[0], step_value
                )
                param_values.append(
                    [
                        (val, val + init_value[1] - init_value[0])
                        for val in interpolated_values
                    ]
                )

        elif isinstance(param_data, list):
            param_values.append(param_data)

    configs = list(itertools.product(*param_values))

    result = []
    for config_values in configs:
        config = dict(zip(param_names, config_values))
        result.append(config)

    return result
