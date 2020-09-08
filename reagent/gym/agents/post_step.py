#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import logging
from typing import Union

import gym
import torch
from reagent.gym.preprocessors import (
    make_replay_buffer_inserter,
    make_replay_buffer_trainer_preprocessor,
)
from reagent.gym.types import PostStep, Transition
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer
from reagent.training.trainer import Trainer

import iml_profiler.api as iml
from reagent.training import rlscope_common

logger = logging.getLogger(__name__)


def add_replay_buffer_post_step(
    replay_buffer: ReplayBuffer, env: gym.Env, replay_buffer_inserter=None
):
    """
    Simply add transitions to replay_buffer.
    """

    if replay_buffer_inserter is None:
        replay_buffer_inserter = make_replay_buffer_inserter(env)

    def post_step(transition: Transition) -> None:
        replay_buffer_inserter(replay_buffer, transition)

    return post_step


def train_with_replay_buffer_post_step(
    replay_buffer: ReplayBuffer,
    env: gym.Env,
    trainer: Trainer,
    training_freq: int,
    batch_size: int,
    gradient_steps: int,
    trainer_preprocessor=None,
    device: Union[str, torch.device] = "cpu",
    replay_buffer_inserter=None,
) -> PostStep:
    """ Called in post_step of agent to train based on replay buffer (RB).
        Args:
            trainer: responsible for having a .train method to train the model
            trainer_preprocessor: format RB output for trainer.train
            training_freq: how many steps in between trains
            batch_size: how big of a batch to sample
    """
    if isinstance(device, str):
        device = torch.device(device)

    if trainer_preprocessor is None:
        trainer_preprocessor = make_replay_buffer_trainer_preprocessor(
            trainer, device, env
        )

    if replay_buffer_inserter is None:
        replay_buffer_inserter = make_replay_buffer_inserter(env)

    _num_steps = 0

    def post_step(transition: Transition) -> None:
        nonlocal _num_steps

        with rlscope_common.iml_prof_operation('replay_buffer_add'):
            replay_buffer_inserter(replay_buffer, transition)

        if _num_steps % training_freq == 0:
            # IML: add gradient_steps hyperparameter to perform multiple gradient updates per step.
            # This matches the behaviour of stable-baselines (gradient_steps) and
            # tf-agents (train_steps_per_iteration) hyperparameters.
            #
            # NOTE: the original TD3 paper (https://arxiv.org/pdf/1802.09477.pdf) does not
            # mention this hyperparameter nor have it in its pseudocode, but using it ensures
            # an "apples-to-apples" comparison of the frameworks.
            # logger.info(f"TRAIN: num_steps % training_freq = 0, num_steps={_num_steps}, training_freq={training_freq}")
            for gradient_step in range(gradient_steps):
                with rlscope_common.iml_prof_operation('train_step'):
                    assert replay_buffer.size >= batch_size
                    train_batch = replay_buffer.sample_transition_batch(batch_size=batch_size)
                    preprocessed_batch = trainer_preprocessor(train_batch)
                    trainer.train(preprocessed_batch)
        _num_steps += 1
        return

    return post_step
