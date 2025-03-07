from .food_gather import VirtualHomeGatherFoodEnvV2
from typing import List, Dict, Any
import gymnasium as gym
from gymnasium.core import ObsType, ActType, SupportsFloat
from random import randint


class CrossVirtualHomeGatherFoodEnvV2(gym.Env):

    def __init__(
            self,
            environment_graph_list: List[Dict[str, Any]]
    ):
        self.basic_env_list = []
        for graph in environment_graph_list:
            self.basic_env_list.append(VirtualHomeGatherFoodEnvV2(graph))

        self.env_count = len(self.basic_env_list)

        if self.env_count < 1:
            raise ValueError("Error: unexpected empty environment graph list")

        self.observation_space = self.basic_env_list[0].observation_space
        self.action_space = self.basic_env_list[0].action_space

        self.selected_env_index = -1

    def reset(
            self,
            *,
            seed: int | None = None,
            options: Dict[str, Any] | None = None,
    ) -> tuple[ObsType, Dict[str, Any]]:
        self.selected_env_index = randint(0, self.env_count - 1)
        return self.basic_env_list[self.selected_env_index].reset()

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        if not 0 <= self.selected_env_index < self.env_count:
            raise ValueError("Error: unexpected environment index, make sure that the environment is correctly reset")
        return self.basic_env_list[self.selected_env_index].step(action)

    def render(self):
        """
        Renders the current state of the environment.
        """
        if not 0 <= self.selected_env_index < self.env_count:
            raise ValueError("Error: unexpected environment index, make sure that the environment is correctly reset")
        return self.basic_env_list[self.selected_env_index].render()

    def close(self):
        """
        Closes the environment and releases any resources.
        """
        for env in self.basic_env_list:
            env.close()
        self.selected_env_index = -1


