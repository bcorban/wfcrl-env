import math
import re
from itertools import product
from typing import Union

from wfcrl.environments.data_cases import *
from wfcrl.interface import FastFarmInterface, FlorisInterface
from wfcrl.multiagent_env import MAWindFarmEnv
from wfcrl.simple_env import WindFarmEnv
from wfcrl.wrappers import AECLogWrapper, LogWrapper

env_pattern = r"(Dec_)*(\w+\d*_)(\w+)"
layout_pattern = r"Turb(\d+)_Row(\d+)"

# Register named layouts
registered_simulators = ["Fastfarm", "Floris"]
registered_layouts = list(named_cases_dictionary.keys())
registered_layouts.extend([f"Turb{n}_Row1_" for n in range(1, 13)])
control_types = ["", "Dec_"]
registered_envs = [
    "".join(env_descs)
    for env_descs in product(control_types, registered_layouts, registered_simulators)
]


def get_default_control(controls):
    default_controls = DefaultControl()
    control_dict = {}
    if "yaw" in controls:
        control_dict["yaw"] = default_controls.yaw
    if "pitch" in controls:
        control_dict["pitch"] = default_controls.pitch
    if "torque" in controls:
        control_dict["torque"] = default_controls.torque
    return control_dict


def get_case(name: str, simulator: str):
    simulator_index = registered_simulators.index(simulator)
    # Check for named case
    if name in named_cases_dictionary:
        case_class = named_cases_dictionary[name][simulator_index]
        return case_class()
    # Else Retrieve environment descriptor in env name
    match = re.match(layout_pattern, name)
    num_turbines = int(match.group(1))
    num_rows = int(match.group(2))
    # At this point, only single rowed envs should remain
    # to be procedurally generated
    assert num_rows == 1
    # Procedurally generate a single row wind farm
    case = FarmRowFastfarm() if simulator_index == 0 else FarmRowFloris()
    case.n_turbines = num_turbines
    case.xcoords = case.get_xcoords(num_turbines)
    case.ycoords = case.get_ycoords(num_turbines)
    case.simul_kwargs["xcoords"] = case.xcoords
    case.simul_kwargs["ycoords"] = case.ycoords
    case.interface_kwargs["simul_kwargs"] = case.simul_kwargs
    return case


def validate_case(env_id, case):
    try:
        assert len(case.xcoords) == len(
            case.ycoords
        ), "xcoords and ycoords layout coordinates must have the same length"
        # TODO: add other checks
    except Exception as e:
        raise ValueError(f"Invalid configuration for case {env_id}: {e}")


def make(env_id: str, controls: Union[dict, list] = ["yaw"], log=True, **env_kwargs):
    """Return a wind farm benchmark environment"""
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not a registered WFCRL benchmark environment.")
    match = re.match(env_pattern, env_id)
    decentralized = match.group(1)
    name = match.group(2)
    simulator = match.group(3)
    case = get_case(name, simulator)
    validate_case(env_id, case)
    env_class = MAWindFarmEnv if decentralized == "Dec_" else WindFarmEnv
    simulator_class = FastFarmInterface if simulator == "Fastfarm" else FlorisInterface
    if not isinstance(controls, dict):
        controls = get_default_control(controls)

    env = env_class(
        interface=simulator_class,
        num_turbines=case.n_turbines,
        controls=controls,
        interface_kwargs=case.interface_kwargs,
        start_iter=math.ceil(case.t_init / case.dt),
        **env_kwargs,
    )

    if log:
        wrapper_class = AECLogWrapper if decentralized == "Dec_" else LogWrapper
        env = wrapper_class(env)

    return env


def list_envs():
    return registered_envs