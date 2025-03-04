from typing import Callable
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def train(
        create_env: Callable[[], gym.Env],
        vec_envs: int,
        total_timesteps: int,
        verbose: int = 0,
):
    train_env = make_vec_env(create_env, n_envs=vec_envs)
    train_env = create_env()
    # model = A2C('MultiInputPolicy', env=train_env, verbose=verbose, n_steps=64, ent_coef=0.01)
    model = PPO('MultiInputPolicy', train_env, verbose=verbose)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    return model

def test_env(create_env: Callable[[], gym.Env]):
    vh_env = create_env()
    vh_env.reset()
    for _ in range(100):
        obs = vh_env.action_space.sample()
        print(obs)
        print('--------------')


def train_ppo_model():
    from src.vh_env.food_gather_env import VirtualHomeGatherFoodEnvV2
    from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication

    YOUR_FILE_NAME = "D:\\programs\\windows_exec.v2.2.4\\VirtualHome.exe"
    comm = UnityCommunication(file_name=YOUR_FILE_NAME)
    comm.reset(0)
    comm.add_character()
    res, g = comm.environment_graph()
    comm.close()
    print(g)

    model = train(
        create_env=lambda: VirtualHomeGatherFoodEnvV2(environment_graph=g, log_level='debug'),
        vec_envs=4,
        total_timesteps=500_000,
        verbose=1
    )
    model.save("../../model/v2_first_test.zip")

def test_ppo_model():
    from src.vh_env.food_gather_env import VirtualHomeGatherFoodEnvV2
    from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
    YOUR_FILE_NAME = "D:\\programs\\windows_exec.v2.2.4\\VirtualHome.exe"
    comm = UnityCommunication(file_name=YOUR_FILE_NAME)
    comm.reset(0)
    comm.add_character()
    res, g = comm.environment_graph()
    comm.close()
    # print(g)

    model = PPO.load("../../model/v2_first_test.zip")
    vh_env = VirtualHomeGatherFoodEnvV2(
        environment_graph=g,
        log_level='warning'
    )

    obs, metadata = vh_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        print(f'type{type(action)}, action:{vh_env.ACTION_LIST[action]}')
        obs, reward, done, _, metadata = vh_env.step(action)
    print(vh_env.get_instruction_list())

def main2():
    from src.vh_env.food_gather_env import VirtualHomeGatherFoodEnv
    from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication

    YOUR_FILE_NAME = "D:\\programs\\windows_exec.v2.2.4\\VirtualHome.exe"
    comm = UnityCommunication(file_name=YOUR_FILE_NAME)
    comm.reset(0)
    comm.add_character()
    res, g = comm.environment_graph()
    comm.close()

    test_env(lambda: VirtualHomeGatherFoodEnv(environment_graph=g, log_level='debug'))

if __name__ == "__main__":
    # train_ppo_model()
    # test_ppo_model()
    pass
