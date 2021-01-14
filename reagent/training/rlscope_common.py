import textwrap

import rlscope.api as rlscope
from reagent.training import rlscope_common

def before_each_iteration(iteration, num_iterations, is_warmed_up=None):
  # GOAL: we only want to call report_progress once we've seen ALL the operations run
  # (i.e., q_backward, q_update_target_network).  This will ensure that the GPU HW sampler
  # will see ALL the possible GPU operations.
  waiting_for = OPERATIONS_AVAILABLE.difference(OPERATIONS_SEEN)
  should_report_progress = len(waiting_for) == 0 and ( is_warmed_up is None or is_warmed_up )

  if rlscope.prof.delay and should_report_progress and not rlscope.prof.tracing_enabled:
    # Entire training loop is now running; enable RL-Scope tracing
    rlscope.prof.enable_tracing()

  if rlscope.prof.debug:
    rlscope.logger.info(textwrap.dedent("""\
        RLS: @ t={iteration}: OPERATIONS_SEEN = {OPERATIONS_SEEN}
          waiting for = {waiting_for}
          is_warmed_up = {is_warmed_up}
        """.format(
      iteration=iteration,
      OPERATIONS_SEEN=OPERATIONS_SEEN,
      waiting_for=waiting_for,
      is_warmed_up=is_warmed_up,
    )).rstrip())
  if should_report_progress:
    OPERATIONS_SEEN.clear()
    rlscope.prof.report_progress(
      percent_complete=iteration/float(num_iterations),
      num_timesteps=iteration,
      total_timesteps=num_iterations)
    if rlscope.prof.tracing_enabled:
      rlscope.logger.info(textwrap.dedent("""\
        RLS: @ t={iteration}: PASS {pass_idx}
        """.format(
        pass_idx=rlscope.prof.pass_idx,
        iteration=iteration,
      )).rstrip())

    # if FLAGS.log_stacktrace_freq is not None and iteration % FLAGS.log_stacktrace_freq == 0:
    #   log_stacktraces()

OPERATIONS_SEEN = set()
OPERATIONS_AVAILABLE = set()

def rlscope_register_operations(operations):
  for operation in operations:
    OPERATIONS_AVAILABLE.add(operation)

def rlscope_prof_operation(operation):
  should_skip = operation not in OPERATIONS_AVAILABLE
  op = rlscope.prof.operation(operation, skip=should_skip)
  if not should_skip:
    OPERATIONS_SEEN.add(operation)
  return op

def rlscope_is_warmed_up():
  """
  Return true once we are executing the full training-loop.

  :return:
  """
  assert OPERATIONS_SEEN.issubset(OPERATIONS_AVAILABLE)
  # can_sample = replay_buffer.can_sample(batch_size)
  # return can_sample and OPERATIONS_SEEN == OPERATIONS_AVAILABLE and num_timesteps > learning_starts
  return OPERATIONS_SEEN == OPERATIONS_AVAILABLE
