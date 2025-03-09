from typing import List, Any, Dict
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


class BaseEvaluator:
    """
    A base evaluator class for evaluating reinforcement learning models.
    """

    def _evaluate_param_check(self, episode: int, max_steps: int):
        """
        Validates evaluation parameters.

        Args:
            episode (int): Number of episodes to run.
            max_steps (int): Maximum steps per episode.

        Raises:
            ValueError: If `episode` is less than 1.
            ValueError: If `max_steps` is negative.
        """
        self.none = None
        if episode < 1:
            raise ValueError('Episode must be greater or equal than 1')

        if max_steps < 0:
            raise ValueError('max_steps must be a positive number')

    def _show_result_list_check(self, result_list: List[Any]):
        """
        Checks if the result list is not empty.

        Args:
            result_list (List[Any]): The list containing evaluation results.

        Raises:
            ValueError: If `result_list` is empty.
        """
        self.none = None
        if len(result_list) == 0:
            raise ValueError('No result found in result list, train a model first')

    def _episode_evaluate(
            self,
            env: gym.Env,
            model: Any,
            episode: int,
            max_steps: int,
            deterministic: bool,
    ) -> Dict[str, float]:
        """
        Evaluates a reinforcement learning model over multiple episodes.

        Args:
            env (gym.Env): The environment to evaluate the model on.
            model (Any): The trained model to be evaluated.
            episode (int): Number of episodes to run.
            max_steps (int): Maximum steps per episode.
            deterministic (bool): Whether to use deterministic actions.

        Returns:
            Dict[str, float]: A dictionary containing mean reward, mean episode length, and success rate.
        """
        self._evaluate_param_check(episode, max_steps)

        success = 0  # Count of successful episodes
        reward = 0  # Total accumulated reward across episodes
        episode_length = 0  # Total number of steps taken across episodes

        for n in range(episode):
            i = 0  # Steps taken in the current episode
            obs, _ = env.reset()  # Reset environment and get initial observation

            for t in range(max_steps):
                action, _states = model.predict(obs, deterministic=deterministic)
                obs, action_reward, done, _, _ = env.step(action)
                i += 1
                reward += action_reward  # Accumulate reward
                if done:  # If episode ends, count as success
                    success += 1
                    break
            episode_length += i  # Accumulate episode length

        return {
            'mean_reward': float(reward) / episode,  # Average reward per episode
            'mean_episode_length': float(episode_length) / episode,  # Average episode length
            'success_rate': float(success) / episode,  # Success rate over episodes
        }


class ModelEvaluatorV1(BaseEvaluator):
    """
    A model evaluator that extends BaseEvaluator to compare multiple models.
    """

    def __init__(
            self,
            models: Dict[str, Any],
            env: gym.Env,
    ):
        """
        Initializes the model evaluator with multiple models and an environment.

        Args:
            models (Dict[str, Any]): A dictionary mapping model names to model instances.
            env (gym.Env): The environment in which the models will be evaluated.

        Raises:
            ValueError: If no models are provided.
        """
        self.name_list = list(models.keys())
        self.model_list = list(models.values())
        self.model_count = len(self.model_list)
        self.env = env
        self.result_list: List[Dict[str, Any]] = []

        if self.model_count < 1:
            raise ValueError('Model count must be greater or equal than 1')

    def evaluate(
            self,
            episode: int = 1,
            max_steps: int = 64,
            deterministic: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Evaluates all models over multiple episodes.

        Args:
            episode (int, optional): Number of episodes to run. Defaults to 1.
            max_steps (int, optional): Maximum steps per episode. Defaults to 64.
            deterministic (bool, optional): Whether to use deterministic actions. Defaults to True.

        Returns:
            List[Dict[str, Any]]: A list of evaluation results for each model.
        """
        self.result_list = []
        for model in self.model_list:
            res = super()._episode_evaluate(model=model, env=self.env, episode=episode, max_steps=max_steps,
                                            deterministic=deterministic)
            self.result_list.append(res)
        return self.result_list

    def show_mean_reward(self):
        """
        Displays a plot of mean rewards for each model.
        """
        super()._show_result_list_check(result_list=self.result_list)
        mean_rewards = [entry['mean_reward'] for entry in self.result_list]
        plt.figure(figsize=(8, 6))
        plt.plot(self.name_list, mean_rewards, marker='o', label='Mean Reward')
        plt.xlabel('Model Name')
        plt.ylabel('Mean Reward')
        plt.title(f'Mean Reward of Model Performance on {self.env.__class__.__name__}')
        plt.legend()
        plt.grid(True)
        plt.show()

    def show_mean_episode_length(self):
        """
        Displays a plot of mean episode length for each model.
        """
        super()._show_result_list_check(result_list=self.result_list)
        mean_lengths = [entry['mean_episode_length'] for entry in self.result_list]
        plt.figure(figsize=(8, 6))
        plt.plot(self.name_list, mean_lengths, marker='o', label='Mean Episode Length')
        plt.xlabel('Model Name')
        plt.ylabel('Mean Episode Length')
        plt.title(f'Mean Episode Length of Model Performance on {self.env.__class__.__name__}')
        plt.legend()
        plt.grid(True)
        plt.show()

    def show_success_rate(self):
        """
        Displays a plot of success rate for each model.
        """
        super()._show_result_list_check(result_list=self.result_list)
        success_rates = [entry['success_rate'] for entry in self.result_list]
        plt.figure(figsize=(8, 6))
        plt.plot(self.name_list, success_rates, marker='o', label='Success Rate')
        plt.xlabel('Model Name')
        plt.ylabel('Success Rate')
        plt.title(f'Success Rate of Model Performance on {self.env.__class__.__name__}')
        plt.legend()
        plt.grid(True)
        plt.show()


class ModelEvaluator(BaseEvaluator):
    """
    A model evaluator that extends BaseEvaluator to compare multiple models.
    """

    def __init__(self, models: Dict[str, Any], env: gym.Env):
        """
        Initializes the model evaluator with multiple models and an environment.

        Args:
            models (Dict[str, Any]): A dictionary mapping model names to model instances.
            env (gym.Env): The environment in which the models will be evaluated.

        Raises:
            ValueError: If no models are provided.
        """
        self.name_list = list(models.keys())
        self.model_list = list(models.values())
        self.model_count = len(self.model_list)
        self.env = env
        self.result_list: List[Dict[str, Any]] = []

        if self.model_count < 1:
            raise ValueError('Model count must be greater or equal than 1')

    def evaluate(self, episode: int = 1, max_steps: int = 64, deterministic: bool = True) -> List[Dict[str, Any]]:
        """
        Evaluates all models over multiple episodes.

        Args:
            episode (int, optional): Number of episodes to run. Defaults to 1.
            max_steps (int, optional): Maximum steps per episode. Defaults to 64.
            deterministic (bool, optional): Whether to use deterministic actions. Defaults to True.

        Returns:
            List[Dict[str, Any]]: A list of evaluation results for each model.
        """
        self.result_list = []
        for model in self.model_list:
            res = super()._episode_evaluate(model=model, env=self.env, episode=episode, max_steps=max_steps,
                                            deterministic=deterministic)
            self.result_list.append(res)
        return self.result_list

    def show_mean_reward(self):
        """
        Displays a bar plot of mean rewards for each model.
        """
        super()._show_result_list_check(result_list=self.result_list)
        mean_rewards = [entry['mean_reward'] for entry in self.result_list]
        plt.figure(figsize=(8, 6))
        plt.bar(self.name_list, mean_rewards, color='skyblue')
        plt.xlabel('Model Name')
        plt.ylabel('Mean Reward')
        plt.title(f'Mean Reward of Model Performance on {self.env.__class__.__name__}')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def show_mean_episode_length(self):
        """
        Displays a bar plot of mean episode length for each model.
        """
        super()._show_result_list_check(result_list=self.result_list)
        mean_lengths = [entry['mean_episode_length'] for entry in self.result_list]
        plt.figure(figsize=(8, 6))
        plt.bar(self.name_list, mean_lengths, color='salmon')
        plt.xlabel('Model Name')
        plt.ylabel('Mean Episode Length')
        plt.title(f'Mean Episode Length of Model Performance on {self.env.__class__.__name__}')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def show_success_rate(self):
        """
        Displays a bar plot of success rate for each model.
        """
        super()._show_result_list_check(result_list=self.result_list)
        success_rates = [entry['success_rate'] for entry in self.result_list]
        plt.figure(figsize=(8, 6))
        plt.bar(self.name_list, success_rates, color='lightgreen')
        plt.xlabel('Model Name')
        plt.ylabel('Success Rate')
        plt.title(f'Success Rate of Model Performance on {self.env.__class__.__name__}')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def show_combined_metrics(self):
        """
        Displays a grouped bar plot comparing mean reward and mean episode length for each model.
        with independent y-axes.
        """
        super()._show_result_list_check(result_list=self.result_list)
        mean_rewards = [entry['mean_reward'] for entry in self.result_list]
        mean_lengths = [entry['mean_episode_length'] for entry in self.result_list]
        x = np.arange(len(self.name_list))
        width = 0.35

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plotting Mean Reward on the first y-axis
        bars1 = ax1.bar(x - width / 2, mean_rewards, width, label='Mean Reward', color='royalblue')
        ax1.set_xlabel('Model Name')
        ax1.set_ylabel('Mean Reward', color='royalblue')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.name_list)
        ax1.tick_params(axis='y', labelcolor='royalblue')

        # Create a second y-axis for Mean Episode Length
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width / 2, mean_lengths, width, label='Mean Episode Length', color='darkorange')
        ax2.set_ylabel('Mean Episode Length', color='darkorange')
        ax2.tick_params(axis='y', labelcolor='darkorange')

        # Title and grid
        ax1.set_title(f'Comparison of Mean Reward and Episode Length on {self.env.__class__.__name__}')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # Legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.show()
