#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import dataclasses
import importlib
import logging
import os
import sys
import re
import textwrap

import pprint
import inspect

import click
from ruamel.yaml import YAML

import iml_profiler.api as iml

from reagent.training import rlscope_hyperparams

@click.group()
def reagent():
    from reagent import debug_on_error

    debug_on_error.start()

    os.environ["USE_VANILLA_DATACLASS"] = "0"

    # setup logging in Glog format, approximately...
    FORMAT = (
        "%(levelname).1s%(asctime)s.%(msecs)03d %(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format=FORMAT, datefmt="%m%d %H%M%S"
    )


def _load_func_and_config_class(workflow):
    # Module name should be configurable
    module_name, func_name = workflow.rsplit(".", 1)

    module = importlib.import_module(module_name)
    # Function name should be configurable
    func = getattr(module, func_name)

    # Import in here so that logging and override take place first
    from reagent.core.configuration import make_config_class

    @make_config_class(func)
    class ConfigClass:
        pass

    return func, ConfigClass


def select_relevant_params(config_dict, ConfigClass):
    return {
        field.name: config_dict[field.name]
        for field in dataclasses.fields(ConfigClass)
        if field.name in config_dict
    }

import collections
import torch
Point = collections.namedtuple('Point', ['x', 'y'])

@torch.jit.script
def total(point):
    # type: (Point) -> Tensor
    return point.x + point.y

def wrap_pytorch():
    # assert 'torch' not in sys.modules

    import torch._C

    # FAIL:
    # assert 'torch' not in sys.modules

    type_map = dict()
    for name in dir(torch._C):
        value = getattr(torch._C, name)
        if str(type(value)) not in type_map:
            type_map[str(type(value))] = []
        type_map[str(type(value))].append({'name': name, 'value': str(value)})
        # logging.info(f"TYPE={type(value)}, name={name}, value={value}")
    logging.info(pprint.pformat(type_map))

    from iml_profiler.profiler import log_stacktrace
    from iml_profiler.profiler import wrap_util
    from iml_profiler.profiler.log_stacktrace import LoggedStackTraces
    # import torch._C._nn

    LoggedStackTraces.wrap_module(torch._C)
    LoggedStackTraces.wrap_module(torch._C._nn)
    def should_wrap(name, func):
        return wrap_util.is_builtin(func)
    LoggedStackTraces.wrap_module(torch.nn.functional, should_wrap=should_wrap)

    # NOTE: This messes up torch.jit.script when we override torch.tensor...
    LoggedStackTraces.wrap_module(torch, should_wrap=should_wrap)
    # Sanity check: we must wrap torch._C._nn BEFORE "import torch"
    import torch
    assert type(torch.nn.functional.avg_pool2d) == log_stacktrace.LoggedCall
    assert type(torch.mean) == log_stacktrace.LoggedCall

    LoggedStackTraces.wrap_func(torch.jit, 'script')
    LoggedStackTraces.wrap_func(torch.jit, 'trace')

@reagent.command(short_help="Run the workflow with config file")
@click.argument("workflow")
@click.argument("config_file", type=click.File("r"))
@iml.click_add_arguments()
def run(workflow, config_file, **kwargs):

    func, ConfigClass = _load_func_and_config_class(workflow)

    # print(ConfigClass.__pydantic_model__.schema_json())
    # At this point, we could load config from a JSON file and create an instance of
    # ConfigClass. Then convert that instance to dict (via .asdict()) and apply to
    # the function

    yaml = YAML(typ="safe")
    config_dict = yaml.load(config_file.read())
    assert config_dict is not None, "failed to read yaml file"
    config_dict = select_relevant_params(config_dict, ConfigClass)
    # NOTE: This is when the algorithm gets instantiated.
    config = ConfigClass(**config_dict)

    # Wrap AFTER @torch.jit.script runs to avoid messing up jit compiling
    # (I think wrapping torch.* messes up the type annotation information?)
    # wrap_pytorch()
    from iml_profiler.profiler import clib_wrap as iml_clib_wrap
    iml_clib_wrap.register_torch()

    model = config.model.value
    model_name = model.__class__.__name__
    algo = None
    if re.search(r'DQN', model_name):
        algo = 'dqn'
    elif re.search(r'SAC', model_name):
        algo = 'sac'
    elif re.search(r'TD3', model_name):
        algo = 'td3'
    else:
        raise RuntimeError(f"Not sure what algo to use for model: {model_name}")
    env = config.env.value.env_name
    iml.handle_click_iml_args(kwargs, directory=kwargs['iml_directory'], reports_progress=True)
    iml.prof.set_metadata({
        'algo': algo,
        'env': env,
    })
    process_name = f'{algo}_run'
    phase_name = process_name

    with iml.prof.profile(process_name=process_name, phase_name=phase_name):
        func(**config.asdict())


@reagent.command(short_help="Run the workflow with stable-baselines/rl-baselines-zoo pre-tuned hyperparameters")
@click.argument("workflow")
@click.option("--algo", required=True)
@click.option("--env", required=True)
@iml.click_add_arguments()
def run_stable_baselines(workflow, algo, env, **kwargs):

    config_dict = rlscope_hyperparams.load_stable_baselines_hyperparams(algo, env)

    func, ConfigClass = _load_func_and_config_class(workflow)

    iml.handle_click_iml_args(kwargs, directory=kwargs['iml_directory'], reports_progress=True)
    iml.prof.set_metadata({
        'algo': algo,
        'env': env,
    })
    process_name = f'{algo}_run'
    phase_name = process_name

    config_dict = select_relevant_params(config_dict, ConfigClass)
    config_dict['log_dir'] = os.path.join(kwargs['iml_directory'], 'tensorboard_log_dir')
    logging.info("config:\n{msg}".format(msg=textwrap.indent(pprint.pformat(config_dict), prefix='  ')))

    # NOTE: This is when the algorithm gets instantiated.
    config = ConfigClass(**config_dict)

    # Wrap AFTER @torch.jit.script runs to avoid messing up jit compiling
    # (I think wrapping torch.* messes up the type annotation information?)
    # wrap_pytorch()
    from iml_profiler.profiler import clib_wrap as iml_clib_wrap
    iml_clib_wrap.register_torch()

    with iml.prof.profile(process_name=process_name, phase_name=phase_name):
        func(**config.asdict())


@reagent.command(short_help="Print JSON-schema of the workflow")
@click.argument("workflow")
def print_schema(workflow):
    func, ConfigClass = _load_func_and_config_class(workflow)

    print(ConfigClass.__pydantic_model__.schema_json())


if __name__ == "__main__":
    reagent()
