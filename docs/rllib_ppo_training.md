# Training SpaceLayoutGym with RLlib PPO

This guide walks through running Proximal Policy Optimization (PPO) with
[RLlib](https://docs.ray.io/en/latest/rllib/index.html) on the
`SpaceLayoutGym` environment using the helper script located at
`training/train_rllib_ppo.py`.

## Prerequisites
- Install the project dependencies listed in `requirements.txt`.
- Ensure the `.env` file in the project root defines
  `HOUSING_DESIGN_ROOT_DIR` so that `LaserWallConfig` can locate
  `default_hps_env.yaml` and related assets.
- Install RLlib (for example `pip install "ray[rllib]==2.*"`).

## Basic usage
```bash
python training/train_rllib_ppo.py \
    --train-iterations 20 \
    --train-batch-size 8192 \
    --sgd-minibatch-size 1024 \
    --learning-rate 5e-4 \
    --clip-param 0.2 \
    --entropy-coeff 0.0 \
    --gamma 0.99 \
    --gae-lambda 0.95
```

During training the script prints the metrics returned by RLlib after each
iteration and periodically saves checkpoints under `./rllib_ppo_results`.
Use `--checkpoint-freq` to control how often checkpoints are persisted.

## Hyper-parameter switches
The following flags surface the most common PPO hyper-parameters:

| Flag | RLlib config key | Description |
| --- | --- | --- |
| `--learning-rate` | `lr` | Adam optimizer step size |
| `--train-batch-size` | `train_batch_size` | Number of collected environment steps per iteration |
| `--sgd-minibatch-size` | `sgd_minibatch_size` | Mini-batch size used in PPO's SGD loop |
| `--num-sgd-iter` | `num_sgd_iter` | Gradient steps taken per PPO iteration |
| `--clip-param` | `clip_param` | Clipping range for the PPO objective |
| `--entropy-coeff` | `entropy_coeff` | Entropy bonus encouraging exploration |
| `--gamma` | `gamma` | Discount factor for future rewards |
| `--gae-lambda` | `lambda` | Generalized Advantage Estimation lambda |

Parallel experience collection can be enabled with `--num-workers` and
`--num-envs-per-worker`. Use `--num-gpus` to allocate GPU resources if your
Ray installation has GPU support.

## Environment overrides
By default the script loads the same configuration as
`LaserWallConfig().get_config()`. To tweak environment level settings
(e.g. number of rooms, reward variant, observation encoding) create a YAML or
JSON file with the desired overrides and pass it via `--env-config`:

```yaml
# save as configs/my_experiment.yaml
n_rooms: 5
rewarding_method_name: ZC_Smooth_Log_Reward
action_masking_flag: true
```

```bash
python training/train_rllib_ppo.py --env-config configs/my_experiment.yaml
```

Any keys provided in the override file are merged into the default
configuration before the environment is created.

## Early stopping and reproducibility
Use the optional `--stop-reward`, `--stop-timesteps`, or `--stop-iters`
arguments to terminate training once a target performance or data budget has
been reached. Passing `--seed` forwards the seed to RLlib and the environment
to improve reproducibility (subject to deterministic backend kernels).

## Outputs
- **Checkpoints**: PPO policy checkpoints written to `--output-dir`.
- **Training logs**: RLlib's per-iteration metrics printed to stdout. You can
  also point `--output-dir` to a Ray Tune results directory to integrate with
dashboard tooling.

