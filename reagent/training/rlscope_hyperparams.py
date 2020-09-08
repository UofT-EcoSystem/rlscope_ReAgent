import os

def load_stable_baselines_hyperparams(algo, env_id, rl_baselines_zoo_dir=None):
  """

  Args:
    rl_baselines_zoo_dir:
      Root directory of this repo:
        https://github.com/araffin/rl-baselines-zoo

  Returns:
  """
  from stable_baselines.iml.hyperparams import load_rl_baselines_zoo_hyperparams

  if rl_baselines_zoo_dir is None:
    rl_baselines_zoo_dir = os.getenv('RL_BASELINES_ZOO_DIR', None)
  zoo_params = load_rl_baselines_zoo_hyperparams(
    rl_baselines_zoo_dir, algo, env_id,
    # TODO: should we "undo" what it builds...?
    build_model_layers=True,
  )

  reagent_params = dict()
  allow_none = set()
  if algo == 'td3':

    model = zoo_params['model']
    policy = model.policy_tf

    model_name = 'TD3'
    reagent_params['env'] = dict()
    reagent_params['env']['Gym'] = dict()
    reagent_params['env']['Gym']['env_name'] = env_id
    reagent_params['model'] = dict()
    # Just pass for now...
    reagent_params['passing_score_bar'] = -99999
    reagent_params['model'][model_name] = dict()
    # Most common value in ReAgent.
    reagent_params['num_eval_episodes'] = 20
    # reagent_params['num_train_episodes'] = ...
    # NOTE: match the behaviour of stable-baselines, where "n_timesteps" in the TD3 implementation refers
    # to the number of [Inference, Simulator] "steps" we perform, NOT the number of gradient updates (i..e, train_step calls).
    # reagent_params['num_train_episodes'] = int(zoo_params['hyperparams']['n_timesteps'])
    # reagent_params['num_train_episodes'] = None
    # allow_none.add('num_train_episodes')

    # Stable-baselines counts the initial transitions towards its total step count.
    reagent_params['num_simulator_steps'] = int(zoo_params['hyperparams']['n_timesteps']) - model.learning_starts
    # reagent_params['num_train_episodes'] = int(zoo_params['hyperparams']['n_timesteps']) // model.train_freq
    reagent_params['replay_memory_size'] = model.buffer_size
    reagent_params['train_after_ts'] = model.learning_starts
    reagent_params['train_every_ts'] = model.train_freq
    reagent_params['gradient_steps'] = model.gradient_steps
    reagent_params['use_gpu'] = True
    model_params = reagent_params['model'][model_name]
    def _mk_network_params(policy, is_actor):
      network_params = {
        'FullyConnected': {
          'activations': ['leaky_relu', 'leaky_relu'],
          # 'exploration_variance': zoo_params['hyperparams']['noise_std'],
          'sizes': policy.layers,
        }
      }
      if is_actor and 'noise_std' in zoo_params['hyperparams']:
        network_params['FullyConnected']['exploration_variance'] = zoo_params['hyperparams']['noise_std']
      return network_params
    model_params['actor_net_builder'] = _mk_network_params(policy, is_actor=True)
    model_params['critic_net_builder'] = _mk_network_params(policy, is_actor=False)
    model_params['eval_parameters'] = {
      'calc_cpe_in_training': False,
    }
    learning_rate = float(zoo_params['hyperparams'].get('learning_rate', 3e-4))
    def _mk_optimizer_params():
      return {
        'Adam': {
          'lr': learning_rate,
        },
      }
    model_params['trainer_param'] = {
      'actor_network_optimizer': _mk_optimizer_params(),
      'q_network_optimizer': _mk_optimizer_params(),
      'minibatch_size': zoo_params['model'].batch_size,
      'delayed_policy_update': model.policy_delay,
      'noise_variance': model.target_policy_noise,
      'noise_clip': model.target_noise_clip,
      'rl': {
        'gamma': model.gamma,
        'target_update_rate': model.tau,
      }
    }

  else:
    raise NotImplementedError(f"Not sure how to load ReAgent hyperparameters from rl-baselines-zoo/stable-baselines parameters for algo={algo}")

  unset_hyperparams = set(
    param for param, param_value in reagent_params.items()
    if param_value is None and param not in allow_none)
  assert len(unset_hyperparams) == 0

  return reagent_params
