from typing import Any, SupportsFloat, Dict

import numpy as np
import gymnasium as gym
from gymnasium.core import ActType, ObsType


class Golf2DEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super().__init__()

        self.pos = np.array([0, 0], dtype=np.int32)
        self.action_space = gym.spaces.Discrete(4)
        # 2D Box
        self.observation_space = gym.spaces.Box(
            low=0,
            high=10,
            shape=(2,),
            dtype=np.int32,
        )

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
        self.pos = np.array([0, 0], dtype=np.int32)
        # self.pos = self.np_random.integers(low=0, high=10, size=(2,))
        return self.pos, {}

    def _get_reward(self) -> int:
        return -abs(self.pos[0] - 5) - abs(self.pos[1] - 5)

    def _is_done(self) -> bool:
        return np.array_equal(self.pos, np.array([5, 5]))

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        if action == 0:
            self.pos[0] -= 1
        elif action == 1:
            self.pos[1] += 1
        elif action == 2:
            self.pos[0] += 1
        elif action == 3:
            self.pos[1] -= 1

        self.pos = np.clip(self.pos, 0, 10)

        reward = self._get_reward()

        done = self._is_done()

        return self.pos, reward, done, False, {}

    def render(self, mode='console'):
        if mode == 'console':
            print(f"Current Position: {self.pos}")

    def close(self):
        pass


class Golf2DBoxEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super().__init__()

        self.pos = np.array([0, 0], dtype=np.int32)
        self.action_space = gym.spaces.Discrete(4)
        # 2D Box
        self.observation_space = gym.spaces.Box(
            low=0,
            high=10,
            shape=(2,),
            dtype=np.int32,
        )

        self.observation_space = gym.spaces.Dict({
            "agent_pos": gym.spaces.Box(low=0, high=10, shape=(2,), dtype=np.int32),
            "magic_box": gym.spaces.Box(low=0, high=10, shape=(3, 2), dtype=np.int32)
        })

        self.chests = self._generate_chests()

    def _generate_chests(self) -> np.ndarray:
        num_chests = np.random.randint(2, 4)
        chests = np.random.randint(low=0, high=11, size=(num_chests, 2))
        return chests

    def _get_obs(self) -> Dict[str, np.ndarray]:
        padded_chests = np.full((3, 2), -1, dtype=np.int32)
        padded_chests[:len(self.chests)] = self.chests
        return {
            'agent_pos': self.pos,
            'magic_box': padded_chests,
        }

    def _get_reward(self) -> int:
        reward = 0
        for i, chest in enumerate(self.chests):
            if np.array_equal(self.pos, chest):
                reward += 10
                self.chests = np.delete(self.chests, i, axis=0)
                break
        return reward

    def _is_done(self) -> bool:
        return len(self.chests) == 0

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore

        self.pos = np.array([0, 0], dtype=np.int32)
        self.chests = self._generate_chests()

        return self._get_obs(), {}

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        if action == 0:
            self.pos[0] -= 1
        elif action == 1:
            self.pos[1] += 1
        elif action == 2:
            self.pos[0] += 1
        elif action == 3:
            self.pos[1] -= 1

        self.pos = np.clip(self.pos, 0, 10)

        reward = self._get_reward()

        done = self._is_done()

        return self._get_obs(), reward, done, False, {}

    def render(self, mode='console'):
        if mode == 'console':
            print(f"Agent Position: {self.pos}")
            print(f"Remaining Chests: {self.chests}")

    def close(self):
        pass

from stable_baselines3 import PPO



if __name__ == '__main__':
    from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication

    YOUR_FILE_NAME = "D:\\programs\\windows_exec.v2.2.4\\VirtualHome.exe"
    script = ['<char0> [walk] <salmon> (328)', '<char0> [grab] <salmon> (328)', '<char0> [walk] <fridge> (306)', '<char0> [open] <fridge> (306)', '<char0> [putin] <salmon> (328) <fridge> (306)', '<char0> [walk] <pie> (320)', '<char0> [grab] <pie> (320)', '<char0> [walk] <chocolatesyrup> (332)', '<char0> [grab] <chocolatesyrup> (332)', '<char0> [walk] <fridge> (306)', '<char0> [putin] <pie> (320) <fridge> (306)', '<char0> [putin] <chocolatesyrup> (332) <fridge> (306)', '<char0> [close] <fridge> (306)']
    comm = UnityCommunication(file_name=YOUR_FILE_NAME)
    comm.reset(0)
    comm.add_character()
    comm.render_script(script, recording=True, skip_animation=False)
    comm.close()

