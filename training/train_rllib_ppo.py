#!/usr/bin/env python3
"""Train PPO on SpaceLayoutGym with RLlib.

This script wires together the configurable :class:`SpaceLayoutGym` environment with
RLlib's PPO implementation. It exposes the most commonly tuned hyper-parameters on
the command line so you can iterate quickly without touching the code.

Example
-------
python training/train_rllib_ppo.py \
    --stop-iters 50 \
    --train-batch-size 4096 \
    --sgd-minibatch-size 512 \
    --learning-rate 3e-4 \
    --clip-param 0.2 \
    --entropy-coeff 0.01 \
    --gamma 0.99 \
    --gae-lambda 0.95

Pass ``--env-config`` with a YAML/JSON file to override any ``LaserWallConfig``
setting (e.g. the number of rooms or reward variant).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - defensive import guard
    raise ModuleNotFoundError(
        "PyYAML is required to parse environment override files. "
        "Install it via `pip install pyyaml`."
    ) from exc

from gym_floorplan.envs.fenv_config import LaserWallConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RLlib PPO on SpaceLayoutGym with configurable hyper-parameters.",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default=None,
        help=(
            "Optional path to a YAML or JSON file containing overrides that will be "
            "passed to LaserWallConfig. Use this to tweak room counts, reward "
            "weights, etc."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./rllib_ppo_results",
        help="Directory where checkpoints and training metrics will be stored.",
    )
    parser.add_argument(
        "--train-iterations",
        type=int,
        default=20,
        help="Maximum number of PPO training iterations to run.",
    )
    parser.add_argument(
        "--stop-iters",
        type=int,
        default=None,
        help="Optional cap on training iterations; defaults to --train-iterations if unset.",
    )
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=None,
        help="Stop once this many environment timesteps have been collected (optional).",
    )
    parser.add_argument(
        "--stop-reward",
        type=float,
        default=None,
        help="Stop early once the mean episode reward reaches this threshold (optional).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Adam optimizer learning rate (RLlib config key: lr).",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=8192,
        help="Number of environment steps to collect before each PPO iteration.",
    )
    parser.add_argument(
        "--sgd-minibatch-size",
        type=int,
        default=1024,
        help="Mini-batch size for the PPO SGD phase.",
    )
    parser.add_argument(
        "--num-sgd-iter",
        type=int,
        default=10,
        help="Number of passes over each train batch when optimizing the surrogate loss.",
    )
    parser.add_argument(
        "--clip-param",
        type=float,
        default=0.2,
        help="Clipping parameter for PPO's surrogate objective.",
    )
    parser.add_argument(
        "--entropy-coeff",
        type=float,
        default=0.0,
        help="Entropy bonus coefficient to encourage exploration.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for future rewards.",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="Generalized Advantage Estimation lambda parameter.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of remote rollout workers (set >0 for parallel sampling).",
    )
    parser.add_argument(
        "--num-envs-per-worker",
        type=int,
        default=1,
        help="How many VectorEnv copies to run inside each RLlib worker.",
    )
    parser.add_argument(
        "--num-gpus",
        type=float,
        default=0,
        help="Fractional number of GPUs to allocate (1.0 == use one full GPU).",
    )
    parser.add_argument(
        "--framework",
        choices=["torch", "tf2"],
        default="torch",
        help="Deep learning framework backend to use for the policy network.",
    )
    parser.add_argument(
        "--disable-exploration",
        action="store_true",
        help="Run PPO in evaluation mode (no stochastic exploration).",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=5,
        help="Save a checkpoint every N training iterations (set 0 to disable).",
    )
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Run Ray in local mode for easier debugging (disables parallelism).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    return parser.parse_args()


def _load_env_overrides(path_str: str | None) -> Dict[str, Any]:
    if path_str is None:
        return {}

    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Environment override file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yml", ".yaml"}:
            data = yaml.safe_load(handle)
        elif path.suffix.lower() == ".json":
            data = json.load(handle)
        else:
            raise ValueError(
                f"Unsupported override format '{path.suffix}'. Use YAML or JSON files."
            )

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Override file must define a mapping of config keys to values.")
    return data


def _build_env_config(overrides: Dict[str, Any]) -> Dict[str, Any]:
    env_config = LaserWallConfig(phase="train", hyper_params=overrides).get_config()
    return env_config


def _ensure_output_dir(path: str) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def main() -> None:
    args = parse_args()

    load_dotenv()  # Enables resolving HOUSING_DESIGN_ROOT_DIR from a .env file if present.

    env_overrides = _load_env_overrides(args.env_config)
    env_config = _build_env_config(env_overrides)

    if args.seed is not None:
        env_config["seed"] = args.seed

    output_dir = _ensure_output_dir(args.output_dir)

    stop_iters = args.stop_iters or args.train_iterations

    ray.init(local_mode=args.local_mode, ignore_reinit_error=True)

    algo_config = (
        PPOConfig()
        .environment(env=env_config["env_name"], env_config=env_config)
        .framework(args.framework)
        .training(
            lr=args.learning_rate,
            train_batch_size=args.train_batch_size,
            sgd_minibatch_size=args.sgd_minibatch_size,
            num_sgd_iter=args.num_sgd_iter,
            clip_param=args.clip_param,
            entropy_coeff=args.entropy_coeff,
            gamma=args.gamma,
            lambda_=args.gae_lambda,
        )
        .rollouts(
            num_rollout_workers=args.num_workers,
            num_envs_per_worker=args.num_envs_per_worker,
            rollout_fragment_length="auto",
        )
        .resources(num_gpus=args.num_gpus)
        .exploration(explore=not args.disable_exploration)
    )

    if args.seed is not None:
        algo_config = algo_config.environment(disable_env_checking=False)
        algo_config = algo_config.training(seed=args.seed)

    algo = algo_config.build()

    timesteps_total = 0
    best_reward = float("-inf")

    try:
        for iteration in range(1, stop_iters + 1):
            result = algo.train()
            timesteps_total = result.get("timesteps_total", timesteps_total)
            best_reward = max(best_reward, result.get("episode_reward_mean", float("-inf")))

            print("\n" + "-" * 80)
            print(f"Iteration {iteration}")
            print(pretty_print(result))

            if args.checkpoint_freq and iteration % args.checkpoint_freq == 0:
                checkpoint = algo.save(output_dir)
                print(f"Checkpoint saved to: {checkpoint}")

            if args.stop_timesteps and timesteps_total >= args.stop_timesteps:
                print(
                    f"Reached stop-timesteps={args.stop_timesteps} after iteration {iteration}."
                )
                break

            mean_reward = result.get("episode_reward_mean")
            if args.stop_reward is not None and mean_reward is not None:
                if mean_reward >= args.stop_reward:
                    print(
                        "Stopping training because mean episode reward reached "
                        f"{mean_reward:.3f} ≥ {args.stop_reward}."
                    )
                    break
    finally:
        if args.checkpoint_freq:
            checkpoint = algo.save(output_dir)
            print(f"Final checkpoint saved to: {checkpoint}")

        algo.stop()
        ray.shutdown()

    print("\nTraining completed.")
    if best_reward != float("-inf"):
        print(f"Best mean episode reward observed: {best_reward:.3f}")
    if args.stop_timesteps:
        print(f"Total timesteps collected: {timesteps_total}")


if __name__ == "__main__":
    main()
