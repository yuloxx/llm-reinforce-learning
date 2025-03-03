import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnv

from typing import Callable, Dict, List, Any
from pathlib import Path
from enum import Enum, auto

import matplotlib.pyplot as plt


from .eval_callback import ModelTrainerCallBack


class ModelTrainerRLAlgo(Enum):
    UNKNOWN = auto()
    PPO = auto()
    A2C = auto()


class ModelTrainer:

    def __init__(
            self,
            algo: ModelTrainerRLAlgo,
            create_env: Callable[[], gym.Env],
    ):
        self.algo = algo
        self.create_env = create_env
        self.result_list = []


    def train(
            self,
            vec_envs: int,
            total_timesteps: int,
            hyperparameters_list: List[Dict[str, Any]],
            policy: str | ActorCriticPolicy = 'MultiInputPolicy',
            model_save_path: Path | None = None,
    ) -> List[Dict[str, Any]]:
        if model_save_path is not None:
            model_save_path = str(model_save_path)

        self.result_list = []
        for hyperparameters in hyperparameters_list:
            train_env = make_vec_env(self.create_env, n_envs=vec_envs)
            learner = self._create_sb3_learner(policy, train_env, hyperparameters)

            callback = ModelTrainerCallBack(
                eval_env=train_env,
                n_steps=learner.n_steps,
                n_vec_env=vec_envs,
                verbose=0,
                best_model_save_path=model_save_path,
            )
            model = learner.learn(total_timesteps=total_timesteps, callback=callback)
            eval_dict = callback.get_model_eval()
            eval_dict['model'] = model
            eval_dict['hyperparameters'] = hyperparameters
            self.result_list.append(eval_dict)

        return self.result_list


    def print_result(
            self,
            target_hyperparameter: List[str] = None,
    ):

        """
        self.result_list: a list of dictionaries containing the evaluation metrics
        Dict[str, Any]: A dictionary with the following keys:
            - "mean_reward" (float or None): The mean reward achieved by the model.
            - "ep_length" (float or None): The average episode length.
            - "success_rate" (float or None): The success rate of the model.
            - "hyperparameters" (Dict[str, Any]): A dictionary with the RL hyperparameters.

        Example:
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        """


    def show_mean_reward_chart(self, target_hyperparameter: str):
        # Mock data
        self.result_list = [
            {'mean_reward': 1020, 'ep_length': 21, 'success_rate': 0.82, 'hyperparameters': {'aaa': 1, 'bbb': 2}},
            {'mean_reward': 838, 'ep_length': 34, 'success_rate': 0.88, 'hyperparameters': {'aaa': 1, 'bbb': 2}},
            {'mean_reward': 980, 'ep_length': 25, 'success_rate': 0.85, 'hyperparameters': {'aaa': 2, 'bbb': 1}},
            {'mean_reward': 1150, 'ep_length': 29, 'success_rate': 0.90, 'hyperparameters': {'aaa': 3, 'bbb': 1}},
            {'mean_reward': 765, 'ep_length': 20, 'success_rate': 0.78, 'hyperparameters': {'aaa': 2, 'bbb': 2}},
            {'mean_reward': 920, 'ep_length': 40, 'success_rate': 0.87, 'hyperparameters': {'aaa': 3, 'bbb': 3}}
        ]
        # Check hyperparameter values for sorting
        for result in self.result_list:
            if target_hyperparameter not in result['hyperparameters']:
                raise ValueError(f"Error: The hyperparameter '{target_hyperparameter}' not found.")

        # Sort the result_list based on the target hyperparameter
        self.result_list.sort(key=lambda entry: entry['hyperparameters'][target_hyperparameter])

        # Now extract the mean rewards after sorting
        mean_rewards = [entry['mean_reward'] for entry in self.result_list]
        # Print out sorted mean_reward values and corresponding hyperparameters
        # for idx, entry in enumerate(self.result_list):
        #     print(
        #         f"Index {idx}: Mean Reward = {entry['mean_reward']}, Hyperparameters = {entry['hyperparameters']}")
        # Plotting
        plt.figure()
        plt.plot(mean_rewards, label='Mean Reward', marker='o', color='b')
        # Annotating each point with corresponding hyperparameters
        for idx, entry in enumerate(self.result_list):
            plt.annotate(f"({entry['hyperparameters'][target_hyperparameter]})",
                         (idx, entry['mean_reward']),
                         textcoords="offset points", xytext=(0, 10), ha='center')
        plt.title(f'Mean Reward Trend on {target_hyperparameter}')
        plt.xlabel('Index')
        plt.ylabel('Mean Reward')
        plt.legend()
        plt.grid(True)
        plt.show()
        # Annotate each point with corresponding hyperparameters
        # for idx, entry in enumerate(self.result_list):
        #     plt.annotate(f"({entry['hyperparameters']})",
        #                  (idx, entry['mean_reward']),
        #                  textcoords="offset points", xytext=(0, 10), ha='center')

    def show_ep_length_chart(self, target_hyperparameter: str):
        # Mock data
        self.result_list = [
            {'mean_reward': 1020, 'ep_length': 21, 'success_rate': 0.82, 'hyperparameters': {'aaa': 1, 'bbb': 2}},
            {'mean_reward': 838, 'ep_length': 34, 'success_rate': 0.88, 'hyperparameters': {'aaa': 1, 'bbb': 2}},
            {'mean_reward': 980, 'ep_length': 25, 'success_rate': 0.85, 'hyperparameters': {'aaa': 2, 'bbb': 1}},
            {'mean_reward': 1150, 'ep_length': 29, 'success_rate': 0.90, 'hyperparameters': {'aaa': 3, 'bbb': 1}},
            {'mean_reward': 765, 'ep_length': 20, 'success_rate': 0.78, 'hyperparameters': {'aaa': 2, 'bbb': 2}},
            {'mean_reward': 920, 'ep_length': 40, 'success_rate': 0.87, 'hyperparameters': {'aaa': 3, 'bbb': 3}}
        ]
        # Check hyperparameter values for sorting
        for result in self.result_list:
            if target_hyperparameter not in result['hyperparameters']:
                raise ValueError(f"Error: The hyperparameter '{target_hyperparameter}' not found.")

        # Sort the result_list based on the target hyperparameter
        self.result_list.sort(key=lambda entry: entry['hyperparameters'][target_hyperparameter])

        # Now extract the episode length after sorting
        ep_length_list = [entry['ep_length'] for entry in self.result_list]
        # Plotting
        plt.figure()
        plt.plot(ep_length_list, label='Episode Length', marker='o', color='b')
        # Annotating each point with corresponding hyperparameters
        for idx, entry in enumerate(self.result_list):
            plt.annotate(f"({entry['hyperparameters'][target_hyperparameter]})",
                         (idx, entry['ep_length']),
                         textcoords="offset points", xytext=(0, 10), ha='center')
        plt.title(f'Episode Length Trend on {target_hyperparameter}')
        plt.xlabel('Index')
        plt.ylabel('Episode Length')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_model(
            self,
            model_index: int,
            model_save_path: Path | None = None,
    ):
        pass


    def _create_sb3_learner(
            self,
            policy: str | ActorCriticPolicy,
            vec_env: VecEnv,
            hyperparameters: Dict[str, Any],
    ) -> PPO | A2C:
        if self.algo == ModelTrainerRLAlgo.PPO:
            return PPO(
                policy=policy,
                env=vec_env,
                **hyperparameters
            )
        elif self.algo == ModelTrainerRLAlgo.A2C:
            return A2C(
                policy=policy,
                env=vec_env,
                **hyperparameters
            )
        else:
            raise ValueError(f'unsupported algo: {self.algo}')


if __name__ == "__main__":
    # train_ppo_model()
    # test_ppo_model()
    pass



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
