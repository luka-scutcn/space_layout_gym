# SpaceLayoutGym Environment

This repository contains an OpenAI Gym compatible environment for space layout design.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/RezaKakooee/space_layout_gym.git
   
   ```

2. Create and activate a Conda environment:
   ```
   conda create -n slg python=3.9
   conda activate slg
   ```

3. Install the environment:
   ```
   cd space_layout_gym
   pip install -e gym-floorplan
   ```

4. Install required packages:
   ```
   pip install -r requirements.txt
   ```

5. Create a `.env` file in the root directory of the project and add the following variable:
   ```
   HOUSING_DESIGN_ROOT_DIR=path/to/space_layout_gym
   ```
   Replace `path/to/space_layout_gym` with the actual path to your space_layout_gym directory.



## Usage

Here's a basic example of how to use the environment:

```python
from gym_floorplan.envs.fenv_config import LaserWallConfig
from gym_floorplan.envs.master_env import SpaceLayoutGym

fenv_config = LaserWallConfig().get_config()
env = SpaceLayoutGym(fenv_config)

obs = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render()
env.close()
```

## Training code

We now include a ready-to-run RLlib PPO training entry point at
`training/train_rllib_ppo.py`. The script exposes common hyper-parameters on the
command line and supports environment overrides through YAML/JSON files. Refer to
[the RLlib PPO training guide](docs/rllib_ppo_training.md) for detailed usage
instructions, including how to customize the environment configuration and manage
checkpoints during training.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@article{kakooee2024reimagining,
  title={Reimagining space layout design through deep reinforcement learning},
  author={Kakooee, Reza and Dillenburger, Benjamin},
  journal={Journal of Computational Design and Engineering},
  pages={qwae025},
  year={2024},
  publisher={Oxford University Press}
}
```