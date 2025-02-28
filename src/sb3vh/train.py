import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from typing import Callable


def train(
        create_env: Callable[[], gym.Env],
        vec_envs: int,
        total_timesteps: int,
        verbose: int = 0,
):
    train_env = make_vec_env(create_env, n_envs=vec_envs)
    model = PPO('MultiInputPolicy', env=train_env, verbose=verbose)
    model.learn(total_timesteps=total_timesteps)
    return model


if __name__ == "__main__":
    from src.sb3vh.vh_env import VirtualHomeGatherFoodEnv
    from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication

    YOUR_FILE_NAME = "D:\\programs\\windows_exec.v2.2.4\\VirtualHome.exe"
    comm = UnityCommunication(file_name=YOUR_FILE_NAME)

    comm.add_character()
    res, g = comm.environment_graph()
    vh_env = VirtualHomeGatherFoodEnv(environment_graph=g)

    model = train(
        create_env=lambda: VirtualHomeGatherFoodEnv(g),
        vec_envs=4,
        total_timesteps=1000,
        verbose=1
    )
    model.save("../model/first_test.zip")

