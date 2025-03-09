from typing import List, Any, Dict
import gymnasium as gym
import matplotlib.pyplot as plt

# Steps
# Rewards
# Success

class BaseEvaluator:

    def _evaluate_param_check(
            self,
            episode: int,
            max_steps: int,
    ):
        self.none = None
        if episode < 1:
            raise ValueError('Episode must be greater or equal than 1')

        if max_steps < 0:
            raise ValueError('max_steps must be a positive number')

    def _show_result_list_check(
            self,
            result_list: List[Any],
    ):
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

        self._evaluate_param_check(episode, max_steps)

        success = 0
        reward = 0
        episode_length = 0

        for n in range(episode):
            i = 0
            obs, _ = env.reset()

            for t in range(max_steps):
                action, _states = model.predict(obs, deterministic=deterministic)
                obs, action_reward, done, _, _ = env.step(action)
                i += 1
                reward += action_reward
                if done:
                    success += 1
                    break
            episode_length += i

        return {
            'mean_reward': float(reward) / episode,
            'mean_episode_length': float(episode_length) / episode,
            'success_rate': float(success) / episode,
        }


class ModelEvaluator(BaseEvaluator):

    def __init__(
            self,
            models: Dict[str, Any],
            env: gym.Env,
    ):
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

        self.result_list = []
        for model in self.model_list:
            res = super()._episode_evaluate(model=model, env=self.env, episode=episode, max_steps=max_steps, deterministic=deterministic)
            self.result_list.append(res)

        return self.result_list

    def show_mean_reward(self):
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


class EnvEvaluator(BaseEvaluator):

    def __init__(
            self,
            model: Any,
            model_name: str,

            env_list: List[gym.Env],
    ):
        self.model = model
        self.model_name = model_name

        self.env_list = env_list
        self.env_count = len(env_list)
        self.name_list = [env.__class__.__name__ for env in self.env_list]

        self.result_list: List[Dict[str, Any]] = []
        if self.env_count < 1:
            raise ValueError('Environment count must be greater or equal than 1')

    def evaluate(
            self,
            episode: int = 1,
            max_steps: int = 64,
    ) -> List[Dict[str, Any]]:

        self.result_list = []
        for env in self.env_list:
            res = super()._episode_evaluate(model=self.model, env = env, episode=episode, max_steps=max_steps)
            self.result_list.append(res)
        return self.result_list

    def show_mean_reward(self):
        super()._show_result_list_check(result_list=self.result_list)
        mean_rewards = [entry['mean_reward'] for entry in self.result_list]
        plt.figure(figsize=(8, 6))
        plt.plot(self.name_list, mean_rewards, marker='o', label='Mean Reward')
        plt.xlabel('Env Name')
        plt.ylabel('Mean Reward')
        plt.title(f'Mean Reward of Model Performance on Envs')
        plt.legend()
        plt.grid(True)
        plt.show()


    def show_mean_episode_length(self):
        super()._show_result_list_check(result_list=self.result_list)
        mean_rewards = [entry['mean_episode_length'] for entry in self.result_list]
        plt.figure(figsize=(8, 6))
        plt.plot(self.name_list, mean_rewards, marker='o', label='Mean Episode Length')
        plt.xlabel('Env Name')
        plt.ylabel('Mean Episode Length')
        plt.title(f'Mean Episode Length of Model Performance on Envs')
        plt.legend()
        plt.grid(True)
        plt.show()

    def show_success_rate(self):
        super()._show_result_list_check(result_list=self.result_list)
        mean_rewards = [entry['success_rate'] for entry in self.result_list]
        plt.figure(figsize=(8, 6))
        plt.plot(self.name_list, mean_rewards, marker='o', label='Success Rate')
        plt.xlabel('Env Name')
        plt.ylabel('Success Rate')
        plt.title(f'Success Rate of Model Performance on Envs')
        plt.legend()
        plt.grid(True)
        plt.show()
