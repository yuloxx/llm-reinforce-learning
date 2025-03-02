import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from typing import Callable, Dict, List, Any
from pathlib import Path
from enum import Enum, auto

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from tqdm import tqdm

class ModelTrainerRLAlgo(Enum):
    UNKNOWN = auto()
    PPO = auto()
    A2C = auto()


class ModelTrainerCallBack(EvalCallback):

    def __init__(
            self,
            n_steps: int,
            n_vec_env: int,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.progress_bar = None
        self.batch_steps = n_steps * n_vec_env

    def _calc_actual_steps(
            self,
            target_steps: int
    ):
        actual_steps = 0
        while target_steps > 0:
            target_steps -= self.batch_steps
            actual_steps += self.batch_steps
        return actual_steps


    def _on_training_start(self) -> None:

        super()._on_training_start()
        # self.progress_bar = tqdm(total=)
        target_steps = self.locals["total_timesteps"] - self.model.num_timesteps
        actual_steps = self._calc_actual_steps(target_steps=target_steps)
        self.progress_bar = tqdm(total=actual_steps)

    def _on_training_end(self) -> None:
        super()._on_training_end()
        self.progress_bar.close()

    def _on_step(self) -> bool:
        res = super()._on_step()
        self.progress_bar.update(self.training_env.num_envs)
        # print(f"training_env: {self.training_env.num_envs}")
        return res


class ModelTrainer:

    def __init__(
            self,
            algo: ModelTrainerRLAlgo,
            create_env: Callable[[], gym.Env],
    ):
        self.algo = algo
        self.create_env = create_env


    def train(
            self,
            vec_envs: int,
            total_timesteps: int,
            hyperparameters_list: List[Dict[str, Any]],
            policy: str | ActorCriticPolicy = 'MultiInputPolicy',
            model_save_path: Path | None = None,
            verbose: int = 0,
    ):
        model = None
        for hyperparameters in hyperparameters_list:
            train_env = make_vec_env(self.create_env, n_envs=vec_envs)
            learner = self._create_sb3_learner(policy, train_env, hyperparameters, verbose)
            callback = ModelTrainerCallBack(eval_env=train_env, n_steps=learner.n_steps, n_vec_env=vec_envs)
            model = learner.learn(total_timesteps=total_timesteps, callback=callback)

        if model_save_path is not None and model is not None:
            model.save(model_save_path)


    def _create_sb3_learner(
            self,
            policy: str | ActorCriticPolicy,
            vec_env: VecEnv,
            hyperparameters: Dict[str, Any],
            verbose: int,
    ) -> PPO | A2C:
        if self.algo == ModelTrainerRLAlgo.PPO:
            return PPO(
                policy=policy,
                env=vec_env,
                verbose=verbose,
                **hyperparameters
            )
        elif self.algo == ModelTrainerRLAlgo.A2C:
            return A2C(
                policy=policy,
                env=vec_env,
                verbose=verbose,
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
