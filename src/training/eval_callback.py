import os
from itertools import accumulate
from typing import Dict, Any

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization
from tqdm import tqdm


class ModelTrainerCallBack(EvalCallback):
    """
    Custom callback that extends the EvalCallback from Stable-Baselines3,
    adding support for progress bar updates and step calculation based on
    the number of steps in a batch and number of environments. This callback
    tracks the performance of the model during training, logs evaluation metrics,
    and saves the best model based on the evaluation results.

    Args:
        n_steps (int): The number of steps per environment to be executed in a batch.
        n_vec_env (int): The number of environments to run in parallel.
        *args: Additional positional arguments passed to the parent class (EvalCallback).
            These arguments correspond to the following parameters from EvalCallback:
                - eval_env: Union[gym.Env, VecEnv]
                - callback_on_new_best: Optional[BaseCallback] = None
                - callback_after_eval: Optional[BaseCallback] = None
                - n_eval_episodes: int = 5
                - eval_freq: int = 10000
                - log_path: Optional[str] = None
                - best_model_save_path: Optional[str] = None
                - deterministic: bool = True
                - render: bool = False
                - verbose: int = 1
                - warn: bool = True
        **kwargs: Additional keyword arguments passed to the parent class (EvalCallback).
    """

    def __init__(
            self,
            n_steps: int,
            n_vec_env: int,
            *args,
            **kwargs
    ):
        """
        Initializes the ModelTrainerCallBack class with the given parameters.

        The parameters from the EvalCallback class will be passed into *args and **kwargs,
        allowing for flexible configuration of the callback.

        Args:
        n_steps (int): The number of steps per environment to be executed in a batch.
        n_vec_env (int): The number of environments to run in parallel.
        *args: Additional positional arguments passed to the parent class (EvalCallback).
            These arguments correspond to the following parameters from EvalCallback:
                - eval_env: Union[gym.Env, VecEnv]
                - callback_on_new_best: Optional[BaseCallback] = None
                - callback_after_eval: Optional[BaseCallback] = None
                - n_eval_episodes: int = 5
                - eval_freq: int = 10000
                - log_path: Optional[str] = None
                - best_model_save_path: Optional[str] = None
                - deterministic: bool = True
                - render: bool = False
                - verbose: int = 1
                - warn: bool = True
        **kwargs: Additional keyword arguments passed to the parent class (EvalCallback).

        """
        super().__init__(*args, **kwargs)
        self.progress_bar = None
        self.batch_steps = n_steps * n_vec_env  # Total steps per update
        self.postfix = {
            "mean_reward": None,
            "ep_length": None,
            "success_rate": None,
            "train/learning_rate": [],
            "train/entropy_loss": [],
            "train/value_loss": [],
        }

    def get_model_eval(self) -> Dict[str, Any]:
        """
        Retrieves the evaluation metrics of the model.

        This method returns a dictionary containing the evaluation metrics, which
        can be used to assess the performance of the model during training. The metrics
        include the mean reward, episode length, and success rate.

        Returns:
        Dict[str, Any]: A dictionary with the following keys:
            - "mean_reward" (float or None): The mean reward achieved by the model.
            - "ep_length" (float or None): The average episode length.
            - "success_rate" (float or None): The success rate of the model.

        Notes:
        - The values in the dictionary may be `None` if the model has not been evaluated yet.
        - This method is typically used after an evaluation step to obtain the latest metrics.
        """

        return self.postfix

    def _calc_actual_steps(self, target_steps: int) -> int:
        """
        Calculates the actual number of steps to take based on the target steps
        and the batch size (steps per batch). The calculation ensures that the steps
        account for the batch configuration.

        Args:
            target_steps (int): The number of steps required to complete the target.

        Returns:
            int: The total number of steps that can be executed based on the batch size.
        """
        actual_steps = 0
        while target_steps > 0:
            target_steps -= self.batch_steps
            actual_steps += self.batch_steps
        return actual_steps

    def _on_training_start(self) -> None:
        """
        This method is triggered when training begins. It initializes the progress bar
        and calculates the number of steps to be performed during the training process.

        The callback records the starting state and sets up the environment.
        """
        self.postfix = {
            "mean_reward": None,
            "ep_length": None,
            "success_rate": None,
            "train/learning_rate": [],
            "train/entropy_loss": [],
            "train/value_loss": [],
            "train/mean_reward": [],
            "train/ep_length": [],
        }
        super()._on_training_start()
        target_steps = self.locals["total_timesteps"] - self.model.num_timesteps
        actual_steps = self._calc_actual_steps(target_steps=target_steps)
        self.progress_bar = tqdm(total=actual_steps)

    def _on_training_end(self) -> None:
        """
        This method is triggered when training ends. It closes the progress bar to
        finalize the tracking of the training progress.
        """
        super()._on_training_end()
        self.progress_bar.close()
        self.postfix["train/mean_reward"] = [
            float(sum_reward) / (i + 1)
            for i, sum_reward in enumerate(accumulate(self.postfix["train/mean_reward"]))
        ]


    def _on_step(self) -> bool:
        """
        This method is called at each step of training. It is responsible for updating
        the progress bar, logging the evaluation metrics, and performing model evaluation.

        It also determines whether the training should continue based on the evaluation results.

        Returns:
            bool: True if training should continue, False otherwise.
        """
        # Update the progress bar based on the number of environments
        self.progress_bar.update(self.training_env.num_envs)

        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval environments if necessary
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval environments are not wrapped the same way. "
                        "Refer to the Stable-Baselines3 documentation for correct setup."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            # Perform evaluation using the provided evaluation environment
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            # Log evaluation results if log_path is provided
            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            # Calculate mean and std for reward and episode length
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            # Log evaluation details
            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, "
                      f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            # Log success rate if available
            success_rate = None
            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")

            # print(self.logger.name_to_value)
            value_loss = self.logger.name_to_value["train/value_loss"]
            entropy_loss = self.logger.name_to_value["train/entropy_loss"]
            learning_rate = self.logger.name_to_value["train/learning_rate"]

            self.postfix["train/learning_rate"].append(learning_rate)
            self.postfix["train/entropy_loss"].append(entropy_loss)
            self.postfix["train/value_loss"].append(value_loss)

            self.postfix["train/mean_reward"].append(mean_reward)
            self.postfix["train/ep_length"].append(mean_ep_length)

            # PPO: dict_keys(['train/learning_rate', 'train/entropy_loss', 'train/policy_gradient_loss', 'train/value_loss', 'train/approx_kl', 'train/clip_fraction', 'train/loss', 'train/explained_variance', 'train/n_updates', 'train/clip_range', 'eval/mean_reward', 'eval/mean_ep_length', 'time/total_timesteps'])
            # A2C: dict_keys(['train/learning_rate', 'train/n_updates', 'train/explained_variance', 'train/entropy_loss', 'train/policy_loss', 'train/value_loss', 'eval/mean_reward', 'eval/mean_ep_length', 'time/total_timesteps'])

            self.logger.dump(self.num_timesteps)

            # Save the best model if the mean reward improves
            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

            # Update the postfix with evaluation metrics for progress bar
            update_kv = {}
            for key, value in {
                "mean_reward": mean_reward,
                "ep_length": mean_ep_length,
                "learning_rate": learning_rate,
                "value_loss": value_loss,
                "entropy_loss": entropy_loss,
            }.items():
                self.postfix[key] = value
                update_kv[key] = value

            if update_kv:
                self.progress_bar.set_postfix(update_kv)

        return continue_training
