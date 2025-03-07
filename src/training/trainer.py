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
    """
    A class for training and evaluating reinforcement learning (RL) models using Stable-Baselines3.

    This class facilitates the training of RL models with different algorithms (PPO, A2C), hyperparameters,
    and environments. It includes methods to visualize the training progress, such as plotting mean reward,
    episode length, and success rate trends based on various hyperparameters.

    Attributes:
        algo (ModelTrainerRLAlgo): The reinforcement learning algorithm (PPO or A2C) to be used.
        create_env (Callable[[], gym.Env]): A callable that generates the training environment.
        result_list (List[Dict[str, Any]]): A list to store training results, including model evaluations
                                             and hyperparameters.

    Methods:
        __init__(algo: ModelTrainerRLAlgo, create_env: Callable[[], gym.Env]):
            Initializes the ModelTrainer with the specified algorithm and environment generator.

        train(vec_envs: int, total_timesteps: int, hyperparameters_list: List[Dict[str, Any]],
              policy: str | ActorCriticPolicy = 'MultiInputPolicy', model_save_path: Path | None = None) -> List[Dict[str, Any]]:
            Trains the model using the specified hyperparameters, policy, and environment.

        show_mean_reward_chart(target_hyperparameter: str):
            Displays a chart showing the mean reward trend based on a target hyperparameter.

        show_ep_length_chart(target_hyperparameter: str):
            Displays a chart showing the episode length trend based on a target hyperparameter.

        show_success_rate_chart(target_hyperparameter: str):
            Displays a chart showing the success rate trend based on a target hyperparameter.

        save_model(model_index: int, model_save_path: Path):
            Saves the model at the specified index in `result_list` to the given file path.
    """

    def __init__(
            self,
            algo: ModelTrainerRLAlgo,
            create_env: Callable[[], gym.Env],
    ):
        """Initializes the ModelTrainer with a specified RL algorithm and an environment creation function.

        Args:
            algo (ModelTrainerRLAlgo): The reinforcement learning algorithm to be used.
                Supported values are:
                - ModelTrainerRLAlgo.UNKNOWN: Default value if the algorithm is not specified.
                - ModelTrainerRLAlgo.PPO: Proximal Policy Optimization.
                - ModelTrainerRLAlgo.A2C: Advantage Actor-Critic.
            create_env (Callable[[], gym.Env]): A callable function that creates and returns a Gym environment instance.
                This function should return a new environment each time it is called.

        Attributes:
            self.algo (ModelTrainerRLAlgo): Stores the selected reinforcement learning algorithm.
            self.create_env (Callable[[], gym.Env]): Stores the function used to create new environment instances.
            self.result_list (list): A list to store results generated during training, such as rewards or evaluation metrics.
        """
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
        """Trains a reinforcement learning model using the specified algorithm and hyperparameters.

        This method trains the model using Stable-Baselines3, supporting multiple sets of hyperparameters.
        It creates vectorized environments, initializes the algorithm, and evaluates the model using a callback.

        Args:
            vec_envs (int): The number of environments to run in parallel.
            total_timesteps (int): The total number of training timesteps.
            hyperparameters_list (List[Dict[str, Any]]): A list of dictionaries containing hyperparameters
                for training the model. Each dictionary should contain parameters specific to the chosen algorithm:

                Example PPO Hyperparameters Copied From https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py:
                    - learning_rate (Union[float, Schedule]): Default is 3e-4.
                    - n_steps (int): Default is 2048.
                    - batch_size (int): Default is 64.
                    - n_epochs (int): Default is 10.
                    - gamma (float): Discount factor, default is 0.99.
                    - gae_lambda (float): Default is 0.95.
                    - clip_range (Union[float, Schedule]): Default is 0.2.
                    - ent_coef (float): Default is 0.0.
                    - vf_coef (float): Default is 0.5.
                    - max_grad_norm (float): Default is 0.5.

                Example A2C Hyperparameters Copied From https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/a2c/a2c.py:
                    - learning_rate (Union[float, Schedule]): Default is 7e-4.
                    - n_steps (int): Default is 5.
                    - gamma (float): Default is 0.99.
                    - gae_lambda (float): Default is 1.0.
                    - ent_coef (float): Default is 0.0.
                    - vf_coef (float): Default is 0.5.
                    - max_grad_norm (float): Default is 0.5.
                    - rms_prop_eps (float): Default is 1e-5.
                    - use_rms_prop (bool): Default is True.
                    - use_sde (bool): Default is False.
                    - sde_sample_freq (int): Default is -1.

            policy (str | ActorCriticPolicy, optional): The policy architecture to use. Defaults to 'MultiInputPolicy'.
            model_save_path (Path | None, optional): Path to save the best model. If None, the model is not saved.

        Returns:
            List[Dict[str, Any]]: A list of evaluation results for each hyperparameter set.
                Each dictionary contains:
                - "model": The trained model instance.
                - "hyperparameters": The hyperparameters used for training.
                - Other evaluation metrics collected during training.

        """

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
                eval_freq=4096,
            )
            model = learner.learn(total_timesteps=total_timesteps, callback=callback)
            eval_dict = callback.get_model_eval()
            eval_dict['model'] = model
            eval_dict['hyperparameters'] = hyperparameters
            self.result_list.append(eval_dict)

        return self.result_list


    def compare_show_final_mean_reward(
            self,
            target_hyperparameter: str
    ):
        """Displays a line chart showing the mean reward trend based on a specified hyperparameter.

        This method visualizes how the mean reward varies with different values of a given hyperparameter.
        The results are sorted by the target hyperparameter before plotting.

        Args:
            target_hyperparameter (str): The hyperparameter to analyze. The results will be sorted based
                on this hyperparameter's values before plotting.

        Raises:
            ValueError: If `self.result_list` is empty (i.e., no training results are available).
            ValueError: If the specified `target_hyperparameter` is not found in the training results.

        Example:
            Suppose `result_list` contains training results for different learning rates:

            ```python
            trainer.show_mean_reward_chart("learning_rate")
            ```

            This will generate a chart showing how the mean reward changes with different learning rates.
        """

        self._show_result_list_check(target_hyperparameter)
        # Sort the result_list based on the target hyperparameter
        self.result_list.sort(key=lambda entry: entry['hyperparameters'][target_hyperparameter])

        # Now extract the mean rewards after sorting
        mean_rewards = [entry['mean_reward'] for entry in self.result_list]

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

    def compare_show_final_ep_length(
            self,
            target_hyperparameter: str
    ):
        """Displays a line chart showing the trend of episode length based on a specified hyperparameter.

        This method visualizes how the episode length varies with different values of a given hyperparameter.
        The results are sorted by the target hyperparameter before plotting.

        Args:
            target_hyperparameter (str): The hyperparameter to analyze. The results will be sorted based
                on this hyperparameter's values before plotting.

        Raises:
            ValueError: If `self.result_list` is empty (i.e., no training results are available).
            ValueError: If the specified `target_hyperparameter` is not found in the training results.

        Example:
            Suppose `result_list` contains training results for different learning rates:

            ```python
            trainer.show_ep_length_chart("learning_rate")
            ```

            This will generate a chart showing how the episode length changes with different learning rates.
        """
        self._show_result_list_check(target_hyperparameter)
        # Sort the result_list based on the target hyperparameter
        self.result_list.sort(key=lambda entry: entry['hyperparameters'][target_hyperparameter])

        # Now extract the episode length after sorting
        ep_length_list = [entry['ep_length'] for entry in self.result_list]
        # Plotting
        plt.figure()
        plt.plot(ep_length_list, label='Episode Length', marker='o', color='g')
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

    def compare_show_final_success_rate(
            self,
            target_hyperparameter: str
    ):
        """Displays a line chart showing the trend of success rate based on a specified hyperparameter.

        This method visualizes how the success rate varies with different values of a given hyperparameter.
        The results are sorted by the target hyperparameter before plotting.

        Args:
            target_hyperparameter (str): The hyperparameter to analyze. The results will be sorted based
                on this hyperparameter's values before plotting.

        Raises:
            ValueError: If `self.result_list` is empty (i.e., no training results are available).
            ValueError: If the specified `target_hyperparameter` is not found in the training results.

        Example:
            Suppose `result_list` contains training results for different batch sizes:

            ```python
            trainer.show_success_rate_chart("batch_size")
            ```

            This will generate a chart showing how the success rate changes with different batch sizes.
        """
        self._show_result_list_check(target_hyperparameter)
        # Sort the result_list based on the target hyperparameter
        self.result_list.sort(key=lambda entry: entry['hyperparameters'][target_hyperparameter])

        # Now extract the success rate after sorting
        ep_length_list = [entry['success_rate'] for entry in self.result_list]
        # Plotting
        plt.figure()
        plt.plot(ep_length_list, label='Success Rate', marker='o', color='r')
        # Annotating each point with corresponding hyperparameters
        for idx, entry in enumerate(self.result_list):
            plt.annotate(f"({entry['hyperparameters'][target_hyperparameter]})",
                         (idx, entry['success_rate']),
                         textcoords="offset points", xytext=(0, 10), ha='center')
        plt.title(f'Episode Success Rate on {target_hyperparameter}')
        plt.xlabel('Index')
        plt.ylabel('Success Rate')
        plt.legend()
        plt.grid(True)
        plt.show()


    def show_learning_rate(
            self,
            target_hyperparameter: str
    ):
        """Displays the learning rate schedule for different values of a specified hyperparameter.

        This method plots the learning rate over training steps for different configurations
        of the specified hyperparameter.

        Args:
            target_hyperparameter (str): The hyperparameter to analyze. The results will be grouped
                by this hyperparameter before plotting.

        Raises:
            ValueError: If `self.result_list` is empty.
            ValueError: If `target_hyperparameter` is not found in the training results.
        """
        self._show_result_list_check(target_hyperparameter)
        learning_rates = [entry['train/learning_rate'] for entry in self.result_list]
        hyperparameter_names = [entry['hyperparameters'][target_hyperparameter] for entry in self.result_list]

        plt.figure(figsize=(8, 6))
        for lr, name in zip(learning_rates, hyperparameter_names):
            plt.plot(lr, marker='o', label=f'Hyperparameter: {name}')

        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule for Different Hyperparameters')
        plt.legend()
        plt.grid(True)
        plt.show()

    def show_entropy_loss(self, target_hyperparameter: str):
        """Displays the entropy loss over training steps for different values of a specified hyperparameter.

        Args:
            target_hyperparameter (str): The hyperparameter to analyze.

        Raises:
            ValueError: If `self.result_list` is empty.
            ValueError: If `target_hyperparameter` is not found in the training results.
        """
        self._show_result_list_check(target_hyperparameter)
        entropy_losses = [entry['train/entropy_loss'] for entry in self.result_list]
        hyperparameter_names = [entry['hyperparameters'][target_hyperparameter] for entry in self.result_list]

        plt.figure(figsize=(8, 6))
        for loss, name in zip(entropy_losses, hyperparameter_names):
            plt.plot(loss, marker='o', label=f'Hyperparameter: {name}')

        plt.xlabel('Step')
        plt.ylabel('Entropy Loss')
        plt.title('Entropy Loss for Different Hyperparameters')
        plt.legend()
        plt.grid(True)
        plt.show()

    def show_value_loss(self, target_hyperparameter: str):
        """Displays the value loss over training steps for different values of a specified hyperparameter.

        Args:
            target_hyperparameter (str): The hyperparameter to analyze.

        Raises:
            ValueError: If `self.result_list` is empty.
            ValueError: If `target_hyperparameter` is not found in the training results.
        """
        self._show_result_list_check(target_hyperparameter)
        value_losses = [entry['train/value_loss'] for entry in self.result_list]
        hyperparameter_names = [entry['hyperparameters'][target_hyperparameter] for entry in self.result_list]

        plt.figure(figsize=(8, 6))
        for loss, name in zip(value_losses, hyperparameter_names):
            plt.plot(loss, marker='o', label=f'Hyperparameter: {name}')

        plt.xlabel('Step')
        plt.ylabel('Value Loss')
        plt.title('Value Loss for Different Hyperparameters')
        plt.legend()
        plt.grid(True)
        plt.show()

    def show_mean_reward(self, target_hyperparameter: str):
        """Displays the mean reward over training steps for different values of a specified hyperparameter.

        Args:
            target_hyperparameter (str): The hyperparameter to analyze.

        Raises:
            ValueError: If `self.result_list` is empty.
            ValueError: If `target_hyperparameter` is not found in the training results.
        """
        self._show_result_list_check(target_hyperparameter)
        mean_rewards = [entry['train/mean_reward'] for entry in self.result_list]
        hyperparameter_names = [entry['hyperparameters'][target_hyperparameter] for entry in self.result_list]

        plt.figure(figsize=(8, 6))
        for reward, name in zip(mean_rewards, hyperparameter_names):
            plt.plot(reward, marker='o', label=f'Hyperparameter: {name}')

        plt.xlabel('Step')
        plt.ylabel('Mean Reward')
        plt.title('Mean Reward for Different Hyperparameters')
        plt.legend()
        plt.grid(True)
        plt.show()

    def show_ep_length(self, target_hyperparameter: str):
        """Displays the episode length over training steps for different values of a specified hyperparameter.

        Args:
            target_hyperparameter (str): The hyperparameter to analyze.

        Raises:
            ValueError: If `self.result_list` is empty.
            ValueError: If `target_hyperparameter` is not found in the training results.
        """
        self._show_result_list_check(target_hyperparameter)
        ep_lengths = [entry['train/ep_length'] for entry in self.result_list]
        hyperparameter_names = [entry['hyperparameters'][target_hyperparameter] for entry in self.result_list]

        plt.figure(figsize=(8, 6))
        for length, name in zip(ep_lengths, hyperparameter_names):
            plt.plot(length, marker='o', label=f'Hyperparameter: {name}')

        plt.xlabel('Step')
        plt.ylabel('Episode Length')
        plt.title('Episode Length for Different Hyperparameters')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_model(
            self,
            model_index: int,
            model_save_path: Path,
    ):
        """Saves a trained model to the specified file path.

        This method retrieves a trained model from `self.result_list` using the provided index
        and saves it to the given file path.

        Args:
            model_index (int): The index of the model in `self.result_list` to be saved.
            model_save_path (Path): The file path where the model will be saved.

        Raises:
            ValueError: If `model_index` is out of the valid range of `self.result_list`.

        Example:
            To save the first trained model to a file named "saved_model.zip":

            ```python
            trainer.save_model(0, Path("saved_model.zip"))
            ```
        """
        result_list_length = len(self.result_list)

        if not 0 <= model_index < result_list_length:
            raise ValueError(f'Model {model_index} is out of range')

        model = self.result_list[model_index]['model']
        model.save(model_save_path)

    def _show_result_list_check(
            self,
            target_hyperparameter: str
    ):
        if len(self.result_list) == 0:
            raise ValueError('No result found in result list, train a model first')

        for result in self.result_list:
            if target_hyperparameter not in result['hyperparameters']:
                raise ValueError(f"Error: The hyperparameter '{target_hyperparameter}' not found.")



    def _create_sb3_learner(
            self,
            policy: str | ActorCriticPolicy,
            vec_env: VecEnv,
            hyperparameters: Dict[str, Any],
    ) -> PPO | A2C:
        """Creates an RL model (PPO or A2C) using Stable-Baselines3 based on the selected algorithm.

        This method initializes a reinforcement learning model with the specified policy, environment,
        and hyperparameters. The type of model (PPO or A2C) is determined by `self.algo`.

        Args:
            policy (str | ActorCriticPolicy): The policy architecture to use for training.
            vec_env (VecEnv): The vectorized environment for training the model.
            hyperparameters (Dict[str, Any]): The hyperparameters to configure the RL algorithm.

        Returns:
            PPO | A2C: An instance of the selected RL algorithm (either PPO or A2C).

        Raises:
            ValueError: If the algorithm specified in `self.algo` is not supported.

        Example:
            ```python
            learner = trainer._create_sb3_learner("MlpPolicy", vec_env, {"learning_rate": 0.0003})
            ```
        """

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

