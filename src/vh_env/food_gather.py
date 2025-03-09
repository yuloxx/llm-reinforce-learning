import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from typing import Any, SupportsFloat, Dict, List
from bidict import bidict
import logging

from transitions.core import Machine, MachineError

from .enums import *


class VirtualHomeGatherFoodEnvV2(gym.Env):
    """
    A custom Gymnasium environment for simulating a food gathering task in a VirtualHome setting.

    This environment simulates a task where an agent interacts with a fridge and food objects in a virtual home.
    The agent's goal is to open the fridge, grab food, and place it into the fridge within a set number of steps.
    The environment provides a state machine where the agent's actions transition between different states based
    on the environment's rules.

    The environment includes the following states:
        - 'fridge_closed_freehand_0', 'fridge_closed_freehand_1', 'fridge_closed_freehand_2': Different states
          of the fridge being closed with varying levels of free hands.
        - 'fridge_open_freehand_0', 'fridge_open_freehand_1', 'fridge_open_freehand_2': Different states
          of the fridge being open, with varying levels of free hands.
        - 'game_over': The state when the game ends.

    Actions available to the agent include:
        - Opening and closing the fridge.
        - Grabbing and placing food.
        - Ending the game.

    The state machine is governed by the following transition rules:
        - **_step_end_game**: Transitions from any state to the 'game_over' state.
        - **_step_open_fridge**: Opens the fridge from a closed state with free hands.
        - **_step_close_fridge**: Closes the fridge from an open state with free hands.
        - **_step_grab_food**: Allows the agent to grab food from the fridge.
        - **_step_place_food**: Allows the agent to place food back into the fridge.

    The environment tracks the following:
        - The state of the fridge (open/closed).
        - The number of free hands the agent has.
        - The state of food objects (whether they are grabbed or placed).
        - The number of steps remaining in the game.

    The `reset()` method initializes the environment and sets the agent's position and food count.
    The `step()` method executes an action, updates the state, and returns the new observation, reward, and done flag.
    The `render()` method (not implemented yet) would visualize the environment.
    The `close()` method is used to clean up and release resources.

    Example usage:
        >>> env = VirtualHomeGatherFoodEnvV2(environment_graph=your_graph, log_level='info')
        >>> observation, metadata = env.reset()
        >>> action = 0  # for example, open fridge
        >>> observation, reward, done, truncated, info = env.step(action)

    """

    @classmethod
    def _set_game_metadata(cls):

        # Reward when the character reaches the target (e.g., a food item or an object)
        cls.TARGET_REACHED_REWARD = 500

        # Reward when the entire food - gathering task is finished
        cls.TASK_FINISH_REWARD = 5000

        # Reward to guide agent to choose proper action
        cls.TASK_GUIDE_REWARD = 10

        # Bonus reward given per step, which can encourage the agent to complete the task in fewer steps
        cls.BONUS_PER_STEP = 30

        # Punishment reward for invalid actions or actions that do not lead to progress
        cls.PUNISHMENT_REWARD = -50

        # The maximum number of steps allowed in a single episode of the game
        cls.MAX_GAME_STEP = 64

        cls.OBSERVATION_SPACE_DTYPE = np.uint8

        cls.FOOD_LIST = [
            'salmon',
            'pie',
            'mincedmeat',
            'juice',
            'pancake',
            'pear',
            'milkshake',
            'salad',
            'wine',
            'chocolatesyrup',
            'chicken',
            'carrot'
        ]

        cls.FOOD_COUNT = len(cls.FOOD_LIST)

        cls.ACTION_LIST = [
            'open_fridge',
            'close_fridge',
            'grab_food',
            'place_food',
            'end_game'
        ]
        cls.ACTION_COUNT = len(cls.ACTION_LIST)

        cls.STATES = [
            'fridge_closed_freehand_0',
            'fridge_closed_freehand_1',
            'fridge_closed_freehand_2',
            'fridge_open_freehand_0',
            'fridge_open_freehand_1',
            'fridge_open_freehand_2',
            'game_over'
        ]

        cls.STATES_COUNT = len(cls.STATES)

        cls.INITIAL_STATE = 'fridge_closed_freehand_2'
        cls.FINISH_STATE = 'game_over'

        cls.TRANSITIONS_RULES = [
            {
                'trigger': '_step_end_game',
                'source': '*',
                'dest': 'game_over'
            },

            {
                'trigger': '_step_open_fridge',
                'source': 'fridge_closed_freehand_1',
                'dest': 'fridge_open_freehand_1',
            },
            {
                'trigger': '_step_open_fridge',
                'source': 'fridge_closed_freehand_2',
                'dest': 'fridge_open_freehand_2'
            },

            {
                'trigger': '_step_close_fridge',
                'source': 'fridge_open_freehand_1',
                'dest': 'fridge_closed_freehand_1'
            },
            {
                'trigger': '_step_close_fridge',
                'source': 'fridge_open_freehand_2',
                'dest': 'fridge_closed_freehand_2'
            },

            {
                'trigger': '_step_grab_food',
                'source': 'fridge_open_freehand_2',
                'dest': 'fridge_open_freehand_1'
            },
            {
                'trigger': '_step_grab_food',
                'source': 'fridge_open_freehand_1',
                'dest': 'fridge_open_freehand_0'
            },
            {
                'trigger': '_step_grab_food',
                'source': 'fridge_closed_freehand_2',
                'dest': 'fridge_closed_freehand_1'
            },
            {
                'trigger': '_step_grab_food',
                'source': 'fridge_closed_freehand_1',
                'dest': 'fridge_closed_freehand_0'
            },

            {
                'trigger': '_step_place_food',
                'source': 'fridge_open_freehand_1',
                'dest': 'fridge_open_freehand_2'
            },
            {
                'trigger': '_step_place_food',
                'source': 'fridge_open_freehand_0',
                'dest': 'fridge_open_freehand_1'
            },

        ]

    def __init__(
            self,
            environment_graph: Dict[str, Any],
            log_level: str = 'info',
    ) -> None:
        """
        Initializes the VirtualHomeGatherFoodEnvV2 environment.

        This environment implements a food-gathering task in VirtualHome with configurable
        action and observation spaces. The agent must collect food items and place them
        into the fridge to complete the task.

        Args:
            environment_graph (Dict[str, Any]):
                The initial environment state graph from VirtualHome. This graph consists
                of nodes (objects, characters) and edges (relationships) that describe the
                environment, including positions, states, and entity interactions.
            log_level (str, optional):
                The logging level for debugging and tracking environment behavior.
                Supported values: 'debug', 'info', 'warning', 'error' (case-insensitive).
                Defaults to "info".

        Attributes:
            environment_graph (Dict[str, Any]):
                Stores the initial environment graph for reference.
            self.observation (Dict[str, Any]):
                Represents the agent's observation of the environment, including:
                    - current_state (int): The current discrete state index.
                    - food_holding (int): Number of food items the agent is holding.
                    - food_state (np.ndarray): Array representing the state of all food items.
                    - target_food_count (int): Number of target food items.
                    - remaining_steps (int): Steps remaining in the episode.
            self.observation_space (gym.spaces.Dict):
                Defines the observation space for the agent, containing:
                    - current_state: Discrete space for the state index.
                    - food_holding: Discrete space (0, 1, or 2 food items).
                    - food_state: Box space representing food item states.
                    - target_food_count: Discrete space for the target food count.
                    - remaining_steps: Discrete space for remaining steps.
                    - fridge_state: Discrete space (0: closed, 1: open).
            self.action_space (gym.spaces.Discrete):
                Defines the available actions the agent can take.
            self.vh_metadata (Dict[str, Any]):
                Stores metadata about the environment, including:
                    - character_id (int): The agent's ID in the virtual home.
                    - food_id_bidict (bidict): Mapping between food names and IDs in the virtual home.
                    - fridge_id (int): The fridge's ID in the virtual home.
                    - instruction_list (List[str]): A list of executable instructions.
                    - fridge_exist_flag (bool): Whether the fridge exists in the environment.
                    - current_place (CharacterPlaceV2Enum): The agent's current location.
            self.state (str):
                The current state of the environment.

        Raises:
            ValueError: If the `log_level` is not a valid logging level.

        Example:
            >>> from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
            >>> PATH_TO_VIRTUALHOME_EXECUTABLE = './path_to_virtualhome.exe'
            >>> comm = UnityCommunication(file_name=PATH_TO_VIRTUALHOME_EXECUTABLE)
            >>> comm.reset(0)
            >>> comm.add_character('Chars/Male1')
            >>> res, g = comm.environment_graph()
            >>> env = VirtualHomeGatherFoodEnvV2(
            ...     environment_graph=g,
            ...     log_level="debug"
            ... )
        """
        self._set_game_metadata()
        super(VirtualHomeGatherFoodEnvV2, self).__init__()

        self._add_logger(log_level)
        self.environment_graph = environment_graph

        self.observation = {
            'current_state': self.STATES.index(self.INITIAL_STATE),
            'food_holding': 0,
            'food_state': np.array([self.FOOD_COUNT], dtype=self.OBSERVATION_SPACE_DTYPE),
            'target_food_count': 0,
            'remaining_steps': self.MAX_GAME_STEP,
        }
        self.observation_space = gym.spaces.Dict({
            'current_state': gym.spaces.Discrete(self.STATES_COUNT),
            'food_holding': gym.spaces.Discrete(3),
            'food_state': gym.spaces.Box(
                low=np.iinfo(self.OBSERVATION_SPACE_DTYPE).min,
                high=np.iinfo(self.OBSERVATION_SPACE_DTYPE).max,
                shape=(self.FOOD_COUNT,),
                dtype=self.OBSERVATION_SPACE_DTYPE,
            ),
            'target_food_count': gym.spaces.Discrete(self.FOOD_COUNT),
            'remaining_steps': gym.spaces.Discrete(self.MAX_GAME_STEP + 1),
            'fridge_state': gym.spaces.Discrete(2)
        })
        self.action_space = gym.spaces.Discrete(self.ACTION_COUNT)

        self.vh_metadata = {
            'character_id': 0,  # ID of the character in the environment
            'food_id_bidict': bidict({}),  # Bidirectional mapping between food names and IDs
            'fridge_id': 0,
            'instruction_list': [],  # Instruction list to execute by virtual home
            'fridge_exist_flag': False,  # If the fridge exist
            'current_place': CharacterPlaceV2Enum.NONE | 0,
        }

        self.is_success = False
        self.state = self.INITIAL_STATE

        Machine(model=self, states=self.STATES, transitions=self.TRANSITIONS_RULES, initial=self.INITIAL_STATE)

    def reset(
            self,
            *,
            seed: int | None = None,
            options: Dict[str, Any] | None = None,
    ) -> tuple[ObsType, Dict[str, Any]]:
        """
        Resets the environment to its initial state and returns the initial observation.

        This method reinitializes the environment by resetting the metadata, observation
        space, and extracting relevant entities (food items, fridge, character) from the
        environment graph. The agent starts in the predefined initial state.

        Args:
            seed (int, optional):
                A seed for random number generation to ensure reproducibility.
                Defaults to None.
            options (Dict[str, Any], optional):
                Additional options for resetting the environment.
                Defaults to None.

        Returns:
            tuple[ObsType, Dict[str, Any]]:
                A tuple containing:
                - **observation (ObsType)**: A dictionary representing the initial environment state.
                - **vh_metadata (Dict[str, Any])**: A dictionary containing metadata about the environment.

        Raises:
            ValueError: If no fridge is found in the environment.
        """

        self.vh_metadata = {
            'character_id': 0,  # ID of the character in the environment
            'food_id_bidict': bidict({}),  # Bidirectional mapping between food names and IDs
            'fridge_id': 0,
            'instruction_list': [],  # Instruction list to execute by virtual home
            'fridge_exist_flag': False,  # If the fridge exist
            'current_place': CharacterPlaceV2Enum.NONE | 0,
        }

        self.observation = {
            'current_state': self.STATES.index(self.INITIAL_STATE),
            'food_holding': 0,
            'food_state': np.zeros((self.FOOD_COUNT,), dtype=self.OBSERVATION_SPACE_DTYPE),
            'target_food_count': 0,
            'remaining_steps': self.MAX_GAME_STEP,
            'fridge_state': 0
        }

        for node in self.environment_graph['nodes']:
            if node['class_name'] in self.FOOD_LIST:
                target_food_index = self.FOOD_LIST.index(node['class_name'])
                self.observation['food_state'][target_food_index] = FoodStateV2Enum.INITIAL | 0
                self.vh_metadata['food_id_bidict'][node['class_name']] = node['id']
                self.observation['target_food_count'] += 1
            if node['class_name'] == 'fridge':
                self.vh_metadata['fridge_exist_flag'] = True
                self.vh_metadata['fridge_id'] = node['id']
            if node['class_name'] == 'character':
                self.vh_metadata['character_id'] = node['id']

        if not self.vh_metadata['fridge_exist_flag']:
            raise ValueError('fridge does not exist')

        self.state = self.INITIAL_STATE
        return self.observation, self.vh_metadata

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Executes a single step in the environment based on the given action.

        This method processes the agent's action, updates the environment state accordingly,
        and calculates the reward. It also determines whether the episode has ended due to
        task completion or exceeding the step limit.

        Args:
            action (ActType):
                The action to be executed. It can be either:
                - An integer index corresponding to an action in `ACTION_LIST`.
                - A string representing the action name.

        Returns:
            tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
                A tuple containing:
                - **observation (ObsType)**: The updated environment state.
                - **reward (SupportsFloat)**: The reward received for executing the action.
                - **is_done (bool)**: Whether the episode has ended.
                - **truncated (bool)**: Whether the episode was truncated (always `False` in this case).
                - **vh_metadata (Dict[str, Any])**: Additional metadata about the environment.

        Raises:
            ValueError: If the provided action is invalid (not an integer within range or an unrecognized string).
        """
        # self.logger.debug(f'[action type] {type(action)}')
        if isinstance(action, (int, np.integer)) and 0 <= action < self.ACTION_COUNT:
            pass
        elif isinstance(action, str):
            action = self.ACTION_LIST.index(action)
        elif isinstance(action, np.ndarray) and action.size == 1:
            action = int(action.item())
        else:
            raise ValueError('invalid action')

        self.observation['remaining_steps'] -= 1
        is_done = self.observation['remaining_steps'] <= 0

        reward = 0
        if action == self.ACTION_LIST.index('open_fridge'):
            reward = self._step_open_fridge_wrapper()
        elif action == self.ACTION_LIST.index('close_fridge'):
            reward = self._step_close_fridge_wrapper()
        elif action == self.ACTION_LIST.index('grab_food'):
            reward = self._step_grab_food_wrapper()
        elif action == self.ACTION_LIST.index('place_food'):
            reward = self._step_place_food_wrapper()
        elif action == self.ACTION_LIST.index('end_game'):
            reward = self._step_end_game_wrapper()
            is_done = True

        self._log_state(self.ACTION_LIST[action], is_done, reward)

        return self.observation, reward, is_done, False, self.vh_metadata

    def render(self):
        """
        Renders the current state of the environment.
        """
        pass  # Implementation to be added

    def close(self):
        """
        Closes the environment and releases any resources.
        """
        pass

    def get_instruction_list(self) -> List[str]:
        """
        Retrieve the complete history of agent-environment interaction commands.

        Returns:
            List[str]: Chronological sequence of executed instructions:
                Example: ['<char0> [walk] <salmon> (328)', '<char0> [putin] <salmon> (328) <fridge> (62)']

        Note:
            The returned list is a direct reference to the metadata storage. For immutability,
            consider creating a copy when needing to modify the results.
        """
        return self.vh_metadata['instruction_list']

    def _log_state(
            self,
            action: str,
            is_done: bool,
            reward: int
    ):
        """
        Logs the current environment state after an action is taken.

        This method records the action, updated environment state,
        and relevant metadata for debugging purposes.
        """
        self.logger.debug('--- environment info ---')
        self.logger.debug(f'action: {action}')
        self.logger.debug(f'is_done: {is_done}')
        self.logger.debug(f'reward: {reward}')
        self.logger.debug(f'state: {self.state}')
        self.logger.debug(f'observation: {self.observation}')
        self.logger.debug(f'vh_metadata: {self.vh_metadata}')
        self.logger.debug('\n')

    def _step_end_game_wrapper(self):
        """
        Handles the logic for ending the game and calculating the final reward.

        This method attempts to transition the environment to the "game_over" state.
        If the transition fails (invalid action), a punishment reward is given.
        Otherwise, it evaluates whether the food collection task is successfully completed
        and calculates the final reward accordingly.

        Returns:
            int: The final reward based on task completion and efficiency.
        """

        # 1. Try transitioning to the 'game_over' state
        try:
            self._step_end_game()  # Attempt to trigger the state transition
        except MachineError:
            return self.PUNISHMENT_REWARD  # If transition fails, return a penalty

        # Update the environment's current state to 'game_over'
        self.observation['current_state'] = self.STATES.index(self.FINISH_STATE)

        # 2. Calculate the final reward
        reward = 0
        placed_food_count = 0

        # Count the number of successfully placed food items
        for food_state in self.observation['food_state']:
            if food_state == FoodStateV2Enum.PLACED:
                placed_food_count += 1

        # Check if all required food is placed and the fridge is closed
        if placed_food_count >= self.observation['target_food_count'] and self.observation['fridge_state'] == 0:
            instruction_len = len(self.vh_metadata['instruction_list'])  # Number of steps taken by the agent

            # Compute reward: Base task completion reward + bonus for efficiency
            reward = self.TASK_FINISH_REWARD + self.BONUS_PER_STEP * (
                    self.observation['remaining_steps'] - instruction_len
            )
        else:
            reward = 0  # No reward if the task is incomplete

        return reward

    def _step_place_food_wrapper(self):

        # 1. Automatically find the food to grab.
        # Return punishment is there is no food in initial position
        if self.observation['food_holding'] <= 0:
            return self.PUNISHMENT_REWARD

        target_food_index = -1
        for i, food_state in enumerate(self.observation['food_state']):
            if food_state == FoodStateV2Enum.HOLD:
                target_food_index = i
                break

        if target_food_index == -1:
            return self.PUNISHMENT_REWARD

        # 2. Try transitions
        try:
            self._step_place_food()
        except MachineError:
            return self.PUNISHMENT_REWARD

        # 3. Calculate instruction according to current place
        instruct = []
        food_class = self.FOOD_LIST[target_food_index]
        food_vh_id = self.vh_metadata['food_id_bidict'][food_class]
        fridge_id = self.vh_metadata['fridge_id']

        if self.vh_metadata['current_place'] != CharacterPlaceV2Enum.FRIDGE:
            instruct.append(f'<char0> [walk] <fridge> ({fridge_id})')
        instruct.append(f'<char0> [putin] <{food_class}> ({food_vh_id}) <fridge> ({fridge_id})')
        self.vh_metadata['instruction_list'] += instruct

        # 4. Execute walking to food, Update current place of agent
        self.vh_metadata['current_place'] = CharacterPlaceV2Enum.FRIDGE | 0

        # 5. Update current state
        self.observation['current_state'] = self.STATES.index(self.state)
        self.observation['food_holding'] -= 1
        self.observation['food_state'][target_food_index] = FoodStateV2Enum.PLACED | 0

        return self.TARGET_REACHED_REWARD

    def _step_grab_food_wrapper(self) -> int:
        """
        Handles the logic for the agent grabbing food in the environment.

        This method first checks if the agent can grab food (i.e., has free hands and there is available food).
        If these conditions are met, it transitions the agent's state, generates the required instructions
        for the VirtualHome simulator, updates the environment state, and provides a guidance reward.

        Returns:
            int: The guidance reward if the action is successful, or a punishment reward if the action is invalid.
        """

        # 1. Check if the agent has a free hand to grab food
        if self.observation['food_holding'] >= 2:
            return self.PUNISHMENT_REWARD  # Cannot grab more than 2 items, return a penalty

        # 2. Automatically find the first available food in the initial state
        target_food_index = -1
        for i, food_state in enumerate(self.observation['food_state']):
            if food_state == FoodStateV2Enum.INITIAL:  # Find the first food item that is not yet grabbed
                target_food_index = i
                break

        if target_food_index == -1:
            return self.PUNISHMENT_REWARD  # No food available to grab, return a penalty

        # 3. Attempt state transition to "grabbing food"
        try:
            self._step_grab_food()  # Execute the state transition in the state machine
        except MachineError:
            return self.PUNISHMENT_REWARD  # If transition fails, return a penalty

        # 4. Generate VirtualHome instructions for walking to the food and grabbing it
        instruct = []
        food_class = self.FOOD_LIST[target_food_index]  # Get food name (e.g., 'apple')
        food_vh_id = self.vh_metadata['food_id_bidict'][food_class]  # Get corresponding VirtualHome ID

        instruct.append(f'<char0> [walk] <{food_class}> ({food_vh_id})')  # Walk to the food
        instruct.append(f'<char0> [grab] <{food_class}> ({food_vh_id})')  # Grab the food
        self.vh_metadata['instruction_list'] += instruct  # Add instructions to execution list

        # 5. Update agent's current place to indicate they are now near food
        self.vh_metadata['current_place'] = CharacterPlaceV2Enum.FOOD | 0

        # 6. Update the environment state
        self.observation['current_state'] = self.STATES.index(self.state)  # Sync state with FSM
        self.observation['food_holding'] += 1  # Increase the count of food held by the agent
        self.observation['food_state'][target_food_index] = FoodStateV2Enum.HOLD | 0  # Mark food as "held"

        return self.TASK_GUIDE_REWARD  # Provide a small reward to encourage correct behavior

    def _step_close_fridge_wrapper(self) -> int:
        """
        Handles the logic for the agent closing the fridge in the environment.

        This method checks the agent's current location and decides whether the agent needs to walk
        to the fridge or if the agent is already in front of the fridge. It then generates the appropriate
        instructions for the VirtualHome simulator, attempts to transition the agent's state to close the fridge,
        and updates the environment state accordingly.

        Returns:
            int: The reward for successfully closing the fridge (0 if no reward or penalty).
        """

        # 1. Attempt to close the fridge and handle any state transition errors
        try:
            self._step_close_fridge()  # Execute the state transition to close the fridge
        except MachineError:
            return self.PUNISHMENT_REWARD  # Return a punishment if the state transition fails

        # 2. Generate VirtualHome instructions based on the agent's current place
        instruct = []
        if self.vh_metadata['current_place'] == CharacterPlaceV2Enum.NONE \
                or self.vh_metadata['current_place'] == CharacterPlaceV2Enum.FOOD:
            # If the agent is not in front of the fridge, generate walking instructions
            instruct.append(f'<char0> [walk] <fridge> ({self.vh_metadata["fridge_id"]})')
        elif self.vh_metadata['current_place'] == CharacterPlaceV2Enum.FRIDGE:
            # If the agent is already in front of the fridge, no need to walk
            pass
        # Generate the instruction to close the fridge
        instruct.append(f'<char0> [close] <fridge> ({self.vh_metadata["fridge_id"]})')

        # Add generated instructions to the list of instructions
        self.vh_metadata['instruction_list'] += instruct

        # 3. Update the agent's current location to reflect that they are now interacting with the fridge
        self.vh_metadata['current_place'] = CharacterPlaceV2Enum.FRIDGE | 0

        # 4. Update the environment's state to reflect that the fridge has been closed
        self.observation['current_state'] = self.STATES.index(self.state)  # Sync state with FSM
        self.observation['fridge_state'] = 0  # Mark the fridge as closed (state 0)

        # No specific reward for closing the fridge, return 0
        return 0

    def _step_open_fridge_wrapper(self) -> int:
        """
        Handles the logic for the agent opening the fridge in the environment.

        This method checks the agent's current location, determines if the agent needs to walk
        to the fridge or is already at the fridge, generates the appropriate instructions for the
        VirtualHome simulator to open the fridge, and updates the environment state accordingly.

        Returns:
            int: The reward for successfully opening the fridge (0 if no reward or penalty).
        """

        # 1. Attempt to open the fridge and handle any state transition errors
        try:
            self._step_open_fridge()  # Execute the state transition to open the fridge
        except MachineError:
            return self.PUNISHMENT_REWARD  # Return a punishment if the state transition fails

        # 2. Generate VirtualHome instructions based on the agent's current place
        instruct = []
        if self.vh_metadata['current_place'] == CharacterPlaceV2Enum.NONE \
                or self.vh_metadata['current_place'] == CharacterPlaceV2Enum.FOOD:
            # If the agent is not in front of the fridge, generate walking instructions
            instruct.append(f'<char0> [walk] <fridge> ({self.vh_metadata["fridge_id"]})')
        elif self.vh_metadata['current_place'] == CharacterPlaceV2Enum.FRIDGE:
            # If the agent is already in front of the fridge, no need to walk
            pass
        # Generate the instruction to open the fridge
        instruct.append(f'<char0> [open] <fridge> ({self.vh_metadata["fridge_id"]})')

        # Add generated instructions to the list of instructions
        self.vh_metadata['instruction_list'] += instruct

        # 3. Update the agent's current location to reflect that they are now interacting with the fridge
        self.vh_metadata['current_place'] = CharacterPlaceV2Enum.FRIDGE | 0

        # 4. Update the environment's state to reflect that the fridge has been opened
        self.observation['current_state'] = self.STATES.index(self.state)  # Sync state with FSM
        self.observation['fridge_state'] = 1  # Mark the fridge as open (state 1)

        # No specific reward for opening the fridge, return 0
        return 0

    def _add_logger(self, log_level: str):
        """
        Initializes the logger for the class with the specified log level.

        Args:
            log_level (str): The desired log level. Supported values are 'debug', 'info', 'warning', 'error'.

        Raises:
            ValueError: If an invalid log level is provided.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        log_level = log_level.lower()
        allowed_levels = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR
        }
        if log_level not in allowed_levels:
            raise ValueError(f"Invalid log_level: {log_level}. Must be one of: {list(allowed_levels.keys())}")
        self.logger.setLevel(allowed_levels[log_level])
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            self.logger.addHandler(console_handler)



class VirtualHomeGatherFoodEnv(gym.Env):
    # Reward when the character reaches the target (e.g., a food item or an object)
    TARGET_REACHED_REWARD = 500

    # Reward when the entire food - gathering task is finished
    TASK_FINISH_REWARD = 5000

    # Bonus reward given per step, which can encourage the agent to complete the task in fewer steps
    BONUS_PER_STEP = 5

    # Reward to guide agent to choose proper action
    TASK_GUIDE_REWARD = 10

    # Punishment reward for invalid actions or actions that do not lead to progress
    PUNISHMENT_REWARD = -10

    # The maximum number of steps allowed in a single episode of the game
    MAX_GAME_STEP = 64

    # Data type used for the observation space, here it is set to 8 - bit unsigned integer
    OBSERVATION_SPACE_DTYPE = np.uint8

    FOOD_LIST = [
        'salmon',
        'pie',
        'mincedmeat',
        'juice',
        'pancake',
        'pear',
        'milkshake',
        'salad',
        'wine',
        'chocolatesyrup',
        'chicken',
        'carrot'
    ]

    # OBJECT_LIST = [
    #     'microwave',
    #     'coffeetable',
    #     'kitchentable',
    #     'wallshelf',
    #     'kitchencounter',
    #     'desk',
    #     'fridge',
    #     'bookshelf',
    #     'stove'
    # ]
    OBJECT_LIST = [
        'microwave',
        'coffeetable',
        'fridge',
        'stove'
    ]

    ACTION_LIST = [
        'none',
        'stop',
        'walk_to_food',
        'walk_to_object',
        'grab',
        'put',
        'putin',
        'open',
        'close'
    ]

    @classmethod
    def get_food_count(cls):
        return len(cls.FOOD_LIST)

    @classmethod
    def get_object_count(cls):
        return len(cls.OBJECT_LIST)

    @classmethod
    def get_action_count(cls):
        return len(cls.ACTION_LIST)

    @classmethod
    def get_food_index_dict(cls):
        food_index_dict = {food: i for i, food in enumerate(cls.FOOD_LIST)}
        return food_index_dict

    @classmethod
    def get_object_index_dict(cls):
        object_index_dict = {single_object: i for i, single_object in enumerate(cls.OBJECT_LIST)}
        return object_index_dict

    def __init__(
            self,
            environment_graph: Dict[str, Any],
            log_level: str = 'info',
    ) -> None:
        """
        Initializes the VirtualHomeGatherFoodEnv environment.

        This environment implements a food gathering task in VirtualHome with configurable
        action and observation spaces. The observation space provides comprehensive information
        about the environment state and relationships between entities.

        Args:
            environment_graph (Dict[str, Any]): Initial environment state graph from Virtual Home
                containing nodes (objects, characters) and edges (relationships). The graph should
                include positions, states, and relationships between entities.
            log_level (str): Logger severity level. Supported values: 'debug', 'info',
                'warning', 'error' (case-insensitive). Defaults to "info".

        Raises:
            ValueError: If invalid log_level is provided or environment_graph has unexpected structure
            TypeError: If environment_graph is not a dictionary

        Explanation:
            1. Initializes base gym.Env class through inheritance
            2. Configures logger with specified severity level
            3. Defines MultiDiscrete action space with three dimensions:
               - Action type selection
               - Primary object (food) interaction
               - Secondary object interaction
            4. Constructs dictionary observation space containing:
               - Entity states (food/object existence)
               - Spatial relationships (character-food/object-object)
               - Inventory/fridge state tracking
            5. Initializes observation buffer and Virtual Home metadata

        Example:
            >>> from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
            >>> PATH_TO_YOUR_VIRTUALHOME_EXECUTABLE = './xxx/virtualhome.exe'
            >>> comm = UnityCommunication(file_name=PATH_TO_YOUR_VIRTUALHOME_EXECUTABLE)
            >>> comm.reset(0)
            >>> comm.add_character('Chars/Male1')
            >>> res, g = comm.environment_graph()
            >>> env = VirtualHomeGatherFoodEnv(
            ...     environment_graph=g,
            ...     log_level="debug"
            ... )
        """

        action_count, food_count, object_count = self.get_action_count(), self.get_food_count(), self.get_object_count()

        super(VirtualHomeGatherFoodEnv, self).__init__()
        self._add_logger(log_level)

        self.none = None  # Placeholder for unused variables
        self.environment_graph = environment_graph  # reset initial environment_graph passed by virtual home communicator

        # Define the action space for the character in Virtual Home:
        # - First dimension: Six action types (included in action_list).
        # - Second dimension: The first operated object (food).
        # - Third dimension: The second operated object (object).
        # Note: If the action is "open", the second dimension has no effect.
        self.action_space = gym.spaces.MultiDiscrete(
            [action_count, food_count, object_count]
        )

        # Define the observation space as a dictionary with the following keys:
        # - 'food_state': Binary state of each food item (exists or not).
        # - 'object_state': Binary state of each object (exists or not).
        # - 'food_character_relation': Relationship between food and the character.
        # - 'object_character_relation': Relationship between objects and the character.
        # - 'food_object_relation': Relationship between objects and food items.
        # - 'food_holding': Number of food hold by the agent (range 0 - 2).
        # - 'food_in_fridge': Number of food put into the fridge

        observation_space_dtype = self.OBSERVATION_SPACE_DTYPE
        observation_space_max = np.iinfo(observation_space_dtype).max
        observation_space_min = np.iinfo(observation_space_dtype).min

        self.observation_space = gym.spaces.Dict({
            'food_state': gym.spaces.Box(
                low=observation_space_min,
                high=observation_space_max,
                shape=(food_count,),
                dtype=observation_space_dtype,
            ),
            'object_state': gym.spaces.Box(
                low=observation_space_min,
                high=observation_space_max,
                shape=(object_count,),
                dtype=observation_space_dtype,
            ),
            'food_character_relation': gym.spaces.Box(
                low=observation_space_min,
                high=observation_space_max,
                shape=(food_count,),
                dtype=observation_space_dtype,
            ),
            'object_character_relation': gym.spaces.Box(
                low=observation_space_min,
                high=observation_space_max,
                shape=(object_count,),
                dtype=observation_space_dtype,
            ),
            'food_object_relation': gym.spaces.Box(
                low=observation_space_min,
                high=observation_space_max,
                shape=(food_count, object_count),
                dtype=observation_space_dtype,
            ),
            'food_holding': gym.spaces.Discrete(3),
            'food_in_fridge': gym.spaces.Discrete(food_count),
        })

        # Initialize observation and metadata
        self.observation = self.observation_space.sample()
        self.vh_metadata = self._process_reset_metadata()

    def reset(
            self,
            *,
            seed: int | None = None,
            options: Dict[str, Any] | None = None,
    ) -> tuple[ObsType, Dict[str, Any]]:
        """
        Reset the environment to an initial state and return the initial observation.

        This method performs the following operations in sequence:
        1. Initialize the environment's random number generator with the given seed
        2. Process environment metadata and graph structure
        3. Validate required objects in the environment
        4. Generate initial observation based on the processed state

        Args:
            seed (int | None, optional): Seed for the environment's random number generator.
                A None value will initialize the RNG without a fixed seed. Defaults to None.
            options (Dict[str, Any] | None, optional): Additional configuration options for
                environment reset. Currently not implemented, reserved for future extensions.
                Defaults to None.

        Returns:
            tuple[ObsType, Dict[str, Any]]: Tuple containing:
                - ObsType: Initial observation of the environment
                - Dict[str, Any]: Metadata dictionary containing environment's state information

        Raises:
            ValueError: If required objects (e.g., fridge) are missing in the environment configuration
        """

        self.vh_metadata = self._process_reset_metadata()
        self._process_environment_graph(self.environment_graph)

        # If the fridge not exist, raise an exception
        if not self.vh_metadata['fridge_exist_flag']:
            raise ValueError("Virtual Home environment fridge does not exist")

        return self.observation, self.vh_metadata

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Executes an action in the environment, which could involve interacting with food or objects.

        Args:
            action (ActType): The action to be executed, represented by a tuple consisting of:
                - action_type (ActionEnum): The type of action to execute (e.g., walking to food, grabbing, etc.)
                - food_index (int): The index of the food item to interact with (if applicable)
                - obj_index (int): The index of the object to interact with (if applicable)

        Returns:
            tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
                A tuple containing:
                - The updated observation after the action.
                - A reward, which can be positive, negative, or zero depending on the action outcome.
                - A boolean indicating whether the episode has ended.
                - A boolean indicating whether the episode was truncated.
                - A dictionary containing additional information (metadata).

        Explanation:
            The method executes the action provided in the `action` argument. Based on the action type, the function:
            - Updates the state of food and object relationships.
            - Computes a reward (positive, negative, or zero).
            - Updates the virtual-home environment metadata (vh_metadata).
            - Returns the updated observation, reward, and metadata.

            The method also handles various interaction types like walking to food, grabbing food, putting food into objects, opening/closing objects, etc.

        """
        self.logger.debug('----start step----')
        self.logger.debug(
            f'action: {self.ACTION_LIST[action[0]]}, food: {self.FOOD_LIST[action[1]]}, object: {self.OBJECT_LIST[action[2]]}')
        res = self._step_wrapper(action)
        self.logger.debug(f'observation: {self.observation}')
        self.logger.debug(f'vh_metadata: {self.vh_metadata}')
        self.logger.debug(f'reward: {res[1]}')
        self.logger.debug(f'is_done: {res[2]}')
        self.logger.debug('----end step----\n')
        if res[2]:
            self.logger.info('episode is finished')
        return res

    def render(self):
        """
        Renders the current state of the environment.
        """
        pass  # Implementation to be added

    def close(self):
        """
        Closes the environment and releases any resources.
        """
        pass

    def get_instruction_list(self) -> List[str]:
        """
        Retrieve the complete history of agent-environment interaction commands.

        Returns:
            List[str]: Chronological sequence of executed instructions:
                Example: ['<char0> [walk] <salmon> (328)', '<char0> [putin] <salmon> (328) <fridge> (62)']

        Note:
            The returned list is a direct reference to the metadata storage. For immutability,
            consider creating a copy when needing to modify the results.
        """
        return self.vh_metadata['instruction_list']

    def _add_logger(self, log_level: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        log_level = log_level.lower()
        allowed_levels = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR
        }
        if log_level not in allowed_levels:
            raise ValueError(f"Invalid log_level: {log_level}. Must be one of: {list(allowed_levels.keys())}")
        self.logger.setLevel(allowed_levels[log_level])
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            self.logger.addHandler(console_handler)

    def _step_wrapper(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Execute environment step by dispatching actions to type-specific handlers.

        Handles step counting and termination checks before delegating to action-specific
        methods (STOP/WALK_TO_FOOD/...). Returns standard (obs, reward, done, truncated, info) tuple.
        """
        action_type, _, _ = action

        # Check if the maximum steps have been reached
        is_done = self.vh_metadata['step'] >= self.MAX_GAME_STEP
        # Increment the step count
        self.vh_metadata['step'] += 1

        res = (self.observation, 0, is_done, False, self.vh_metadata)

        if action_type == ActionEnum.NONE:
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        elif action_type == ActionEnum.STOP:
            # Task not finished
            if self.observation['food_in_fridge'] < self.vh_metadata['food_exist_count']:
                return self.observation, self.PUNISHMENT_REWARD, True, False, self.vh_metadata
            # Check if the fridge closed:
            object_index_dict = self.get_object_index_dict()
            object_fridge_index = object_index_dict['fridge']
            # remaining steps:
            remaining_steps = self.MAX_GAME_STEP - self.vh_metadata['step']
            if not self.observation['object_state'][object_fridge_index] & ObjectStateBitmapEnum.OPEN:
                # Fridge closed:
                finial_reward = self.TASK_FINISH_REWARD + remaining_steps * self.BONUS_PER_STEP
                return self.observation, finial_reward, True, False, self.vh_metadata
            else:
                finial_reward = remaining_steps * self.BONUS_PER_STEP
                return self.observation, finial_reward, True, False, self.vh_metadata

        elif action_type == ActionEnum.WALK_TO_FOOD:
            res = self._action_walk_to_food(action, is_done)

        elif action_type == ActionEnum.WALK_TO_OBJECT:
            res = self._action_walk_to_object(action, is_done)

        elif action_type == ActionEnum.GRAB:
            res = self._action_grab(action, is_done)

        elif action_type == ActionEnum.PUT:
            res = self._action_put(action, is_done)

        elif action_type == ActionEnum.PUTIN:
            res = self._action_putin(action, is_done)

        elif action_type == ActionEnum.OPEN:
            res = self._action_open(action, is_done)

        elif action_type == ActionEnum.CLOSE:
            res = self._action_close(action, is_done)

        return res

    def _action_walk_to_food(
            self,
            action: ActType,
            is_done: bool,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Handles the action where the character walks towards a specific food item.

        Args:
            action (ActType): A tuple containing the action type, food index, and object index.
                - The food_index specifies which food item the character will walk to.
            is_done (bool): A boolean flag indicating whether the environment has reached a terminal state (end of the episode).

        Returns:
            tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
                - The updated observation after the character walks to the food.
                - A reward value (always 0, as no reward is given for just walking).
                - A boolean indicating whether the episode has ended (based on the step limit).
                - A boolean indicating whether the episode was truncated (in this case, False).
                - A dictionary containing additional metadata, including updated instruction list.

        Explanation:
            This method processes the action of the character walking towards a specified food item.
            The following steps are executed:

            1. **Check for food existence**: If the specified food item does not exist in the environment (as determined by its state in `food_state`), the character receives a negative reward.
            2. **Reset object-character relations**: Clear any existing object-character relations for the current state, as the action involves moving towards food, not interacting with objects.
            3. **Update food-character relations**:
                - The character's relation to the target food item is set to `CLOSE_TO` (indicating proximity).
                - If the character is holding any other food item, its relation is reset to `NONE`.
            4. **Instruction update**: The action generates a step instruction that is appended to the `instruction_list` in the environment metadata (`vh_metadata`).

            The function is primarily concerned with updating the state of food and character interactions, and the reward is 0 as no significant action (like grabbing or interacting with food) takes place during the walking phase.
        """

        action_count, food_count, object_count = self.get_action_count(), self.get_food_count(), self.get_object_count()
        _, food_index, obj_index = action  # Unpack the action tuple: action_type, food_index, obj_index.

        # Check if the target food item exists in the environment
        if not self.observation['food_state'][food_index] & FoodStateBitmapEnum.EXIST:
            # If the food doesn't exist (its 'EXIST' flag is not set), apply a negative reward
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # Character is walking towards the food; begin by clearing object-character relations
        # Loop over all objects in the environment and reset the character's interaction state with each one
        for i in range(object_count):
            self.observation['object_character_relation'][
                i] = ObjectCharacterStateEnum.NONE | 0  # Reset all object relations to NONE (no interaction)

        # Clear all food-character relations, except for the food item that the character is holding
        # Loop through all food items in the environment
        for i in range(food_count):
            if i == food_index:
                # Set the relation of the character to the target food to CLOSE_TO, indicating proximity
                self.observation['food_character_relation'][i] = FoodCharacterStateEnum.CLOSE_TO | 0
                continue  # Skip this food item since it's the one we're walking to
            if self.observation['food_character_relation'][i] == FoodCharacterStateEnum.HOLD:
                # If the character is holding another food item, skip
                continue
            self.observation['food_character_relation'][i] = FoodCharacterStateEnum.NONE | 0

        # Create a virtual-home instruction based on the current action
        instruction = self._process_step_instruction(action)

        # Append this instruction to the instruction list in the environment metadata
        self.vh_metadata['instruction_list'].append(instruction)

        reward = 0
        if self.observation['food_holding'] < 2 \
                and self.observation['food_character_relation'][food_index] != FoodCharacterStateEnum.CLOSE_TO \
                and self.observation['food_character_relation'][food_index] != FoodCharacterStateEnum.HOLD:
            reward = self.TASK_GUIDE_REWARD

        # Return the updated observation, with no reward for the walking action, and the updated metadata
        return self.observation, reward, is_done, False, self.vh_metadata

    def _action_walk_to_object(
            self,
            action: ActType,
            is_done: bool,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:

        """
        Executes the action of the character walking to an object in the environment.

        Args:
            action (ActType): A tuple containing the action type, food index, and object index.
                - action_type (ActionEnum): The type of action to execute (in this case, walking to an object).
                - food_index (int): The index of the food item to interact with (not relevant for this action).
                - obj_index (int): The index of the object to walk to.

            is_done (bool): A boolean indicating whether the episode has already ended or not.

        Returns:
            tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
                A tuple containing:
                - The updated observation after the action is executed.
                - A reward, which is 0 in this case (no immediate reward for walking to an object).
                - A boolean indicating whether the episode has ended.
                - A boolean indicating whether the episode was truncated.
                - A dictionary containing metadata (vh_metadata), which includes the list of instructions.

        Explanation:
            This function handles the action of the character walking to a specific object. It first checks whether the
            object exists in the environment. If the object does not exist, the function returns a punishment reward.
            If the object exists, it updates the state of the environment to reflect that the character is close to the object
            and clears any prior object-food or object-character relationships. It also appends the action instruction to
            the virtual home metadata.
        """

        action_count, food_count, object_count = self.get_action_count(), self.get_food_count(), self.get_object_count()
        _, food_index, obj_index = action  # Unpack the action tuple: action_type, food_index, obj_index.

        # Check if the object exists in the environment. If not, give a negative reward.
        if not self.observation['object_state'][obj_index] & ObjectStateBitmapEnum.EXIST:
            # Object does not exist, return the observation with a punishment reward
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # If object exists, proceed with walking to the object.

        # Clear all existing relationships between the character and objects
        for i in range(object_count):
            if i == obj_index:
                # Set the 'CLOSE_TO' relation for the target object
                self.observation['object_character_relation'][i] = ObjectCharacterStateEnum.CLOSE_TO | 0
            else:
                # Reset the relation for other objects
                self.observation['object_character_relation'][i] = ObjectCharacterStateEnum.NONE | 0

        # Clear all the food relationships with the character except for the food that might be held in hand
        for i in range(food_count):
            if self.observation['food_character_relation'][i] == FoodCharacterStateEnum.HOLD:
                continue  # Skip if the food is being held by the character
            # Reset any other food-character relations
            self.observation['food_character_relation'][i] = FoodCharacterStateEnum.NONE | 0

        # Generate and append the virtual-home instruction for this step
        instruction = self._process_step_instruction(action)
        self.vh_metadata['instruction_list'].append(instruction)

        reward = -1
        if self.observation['object_character_relation'][obj_index] != ObjectCharacterStateEnum.CLOSE_TO \
                and self.OBJECT_LIST[obj_index] == 'fridge':
            if self.observation['food_holding'] == 0:
                reward = self.TASK_GUIDE_REWARD
            elif self.observation['food_holding'] == 1:
                reward = self.TASK_GUIDE_REWARD
            else:
                object_index = self.get_object_index_dict()['fridge']
                if self.observation['object_state'][object_index] & ObjectStateBitmapEnum.EXIST \
                        and self.observation['object_state'][object_index] & ObjectStateBitmapEnum.CAN_OPEN \
                        and self.observation['object_state'][object_index] & ObjectStateBitmapEnum.OPEN:
                    reward = self.TASK_GUIDE_REWARD
                else:
                    reward = self.PUNISHMENT_REWARD

        # Return the updated observation, with no immediate reward for walking to an object
        return self.observation, reward, is_done, False, self.vh_metadata

    def _action_grab(
            self,
            action: ActType,
            is_done: bool,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Executes the action of the character walking to and grabbing a specific food item in the environment.

        Args:
            action (ActType): A tuple containing the action type, food index, and object index.
                - action_type (ActionEnum): The type of action to execute (in this case, grabbing a food item).
                - food_index (int): The index of the food item the character is trying to grab.
                - obj_index (int): The index of the object (not used for this action, but part of the action tuple).

            is_done (bool): A boolean indicating whether the episode has already ended.

        Returns:
            tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
                A tuple containing:
                - The updated observation after the action is executed.
                - A reward, which is 0 if the action was successful, or a punishment reward if the action failed.
                - A boolean indicating whether the episode has ended.
                - A boolean indicating whether the episode was truncated.
                - A dictionary containing metadata (`vh_metadata`), including the list of instructions.

        Explanation:
            This function handles the action of the character walking to a specific food item and grabbing it.
            - First, it checks if the food exists in the environment. If not, a punishment reward is returned.
            - It then checks if the character is holding more than one food item already; if so, the grab action is not allowed.
            - Next, it checks if the character is close to the target food. If the character is not close, a punishment is applied.
            - If the food is inside an object and that object is closed, a punishment is returned.
            - Finally, if all conditions are satisfied, the character successfully grabs the food, updates the relationships, and returns the updated observation and metadata.
        """

        action_count, food_count, object_count = self.get_action_count(), self.get_food_count(), self.get_object_count()
        _, food_index, obj_index = action  # Unpack the action tuple: action_type, food_index, obj_index.

        # Check if the target food exists in the environment. If not, return a punishment reward.
        if not self.observation['food_state'][food_index] & FoodStateBitmapEnum.EXIST:
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # Check if the character is already holding two food items. If so, return a punishment.
        if self.observation['food_holding'] >= 2:
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # Check if the character is close to the target food. If not, return a punishment.
        if self.observation['food_character_relation'][food_index] != FoodCharacterStateEnum.CLOSE_TO:
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # Check if the food is inside an object and whether the object is open.
        # If the food is inside a closed object, return a punishment.
        none_close_object_index = -1
        for i in range(object_count):
            if self.observation['food_object_relation'][food_index][i] == FoodObjectStateEnum.INSIDE:
                none_close_object_index = i

        # If food is inside a closed object, return punishment
        if none_close_object_index != -1 and \
                not self.observation['object_state'][none_close_object_index] & ObjectStateBitmapEnum.OPEN:
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # If all checks pass, hold the target food in the character's hand.

        # Clear the food-object relations for the target food.
        for i in range(object_count):
            self.observation['food_object_relation'][food_index][i] = FoodObjectStateEnum.NONE | 0

        # Set the food-character relation to `HOLD` for the target food.
        for i in range(food_count):
            if i == food_index:
                self.observation['food_character_relation'][i] = FoodCharacterStateEnum.HOLD | 0
                continue
            if self.observation['food_character_relation'][i] == FoodCharacterStateEnum.HOLD:
                continue  # Skip if the food is being held by the character
            self.observation['food_character_relation'][i] = FoodCharacterStateEnum.NONE | 0

        self.observation['food_holding'] += 1

        # Append the action instruction to the virtual-home instruction list.
        instruction = self._process_step_instruction(action)
        self.vh_metadata['instruction_list'].append(instruction)

        agent_reward = self._process_food_in_fridge()

        # Return the updated observation with no immediate reward for successfully grabbing the food.
        return self.observation, agent_reward, is_done, False, self.vh_metadata

    def _action_put(
            self,
            action: ActType,
            is_done: bool,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Executes the action of putting a food item onto a specific object in the environment.

        Args:
            action (ActType): A tuple containing the action type, food index, and object index.
                - action_type (ActionEnum): The type of action to execute (in this case, putting food on an object).
                - food_index (int): The index of the food item the character is trying to put down.
                - obj_index (int): The index of the object where the food is to be placed.

            is_done (bool): A boolean indicating whether the episode has already ended.

        Returns:
            tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
                A tuple containing:
                - The updated observation after the action is executed.
                - A reward, which is 0 if the action was successful, or a punishment reward if the action failed.
                - A boolean indicating whether the episode has ended.
                - A boolean indicating whether the episode was truncated.
                - A dictionary containing metadata (`vh_metadata`), including the list of instructions.

        Explanation:
            This function handles the action of the character putting a food item onto a specific object in the environment.
            - First, it checks if the character is currently holding any food. If not, a punishment reward is returned.
            - It verifies that the food and object exist in the environment. If either is missing, a punishment is applied.
            - It checks if the character is holding the target food and is close to the target object. If any condition fails, a punishment is given.
            - If all conditions pass, the food is placed on the object, and the relationships between food and object are updated accordingly.
            - The function then appends the action instruction to the virtual-home instruction list and returns the updated state.
        """

        _, food_index, obj_index = action  # Unpack the action tuple: action_type, food_index, obj_index.

        # Check if the character is not holding any food (nothing to put). If true, return a punishment reward.
        if self.observation['food_holding'] <= 0:
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # Check if the target food exists in the environment. If it doesn't exist, return a punishment.
        if not self.observation['food_state'][food_index] & FoodStateBitmapEnum.EXIST:
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # Check if the target object exists in the environment. If it doesn't exist, return a punishment.
        if not self.observation['object_state'][obj_index] & ObjectStateBitmapEnum.EXIST:
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # Check if the character is holding the target food. If not, return a punishment.
        if not self.observation['food_character_relation'][food_index] == FoodCharacterStateEnum.HOLD:
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # Check if the character is close to the target object. If not, return a punishment.
        if not self.observation['object_character_relation'][obj_index] == ObjectCharacterStateEnum.CLOSE_TO:
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # If all checks pass, the character is putting the food on the object.

        # Update the food-character relation: change from 'HOLD' to 'CLOSE_TO' for the target food.
        self.observation['food_character_relation'][food_index] = FoodCharacterStateEnum.CLOSE_TO | 0

        # Update the food-object relation: set the relation to 'ON' for the target food and object.
        self.observation['food_object_relation'][food_index][obj_index] = FoodObjectStateEnum.ON | 0

        self.observation['food_holding'] -= 1

        # Process and append the virtual-home instruction for this action to the instruction list.
        instruction = self._process_step_instruction(action)
        self.vh_metadata['instruction_list'].append(instruction)

        # Return the updated observation, with no immediate reward for successfully putting the food on the object.
        return self.observation, 0, is_done, False, self.vh_metadata

    def _action_putin(
            self,
            action: ActType,
            is_done: bool,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Executes the action of putting a food item into a specific object (e.g., fridge) in the environment.

        Args:
            action (ActType): A tuple containing the action type, food index, and object index.
                - action_type (ActionEnum): The type of action to execute (in this case, putting food into an object).
                - food_index (int): The index of the food item the character is trying to put into the object.
                - obj_index (int): The index of the object (e.g., fridge) where the food is to be placed.

            is_done (bool): A boolean indicating whether the episode has already ended.

        Returns:
            tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
                A tuple containing:
                - The updated observation after the action is executed.
                - A reward, which is calculated based on whether the food was successfully put inside the target object.
                - A boolean indicating whether the episode has ended.
                - A boolean indicating whether the episode was truncated.
                - A dictionary containing metadata (`vh_metadata`), including the list of instructions.

        Explanation:
            This function handles the action of the character putting a food item into a specific object (like a fridge).
            - It first checks if the character is holding food. If not, a punishment reward is given.
            - It checks if the food and object exist in the environment and if the object can be opened. If any condition fails, a punishment is applied.
            - It verifies that the character is holding the food and is close to the target object.
            - If all conditions pass, the food is placed inside the object (e.g., inside a fridge).
            - A reward is given if the food is successfully put into the fridge, based on the number of food items inside the fridge.
            - The function then appends the action instruction to the virtual-home instruction list and returns the updated state.
            - If all food have been successfully placed inside the fridge, end the episode.
        """

        _, food_index, obj_index = action  # Unpack the action tuple: action_type, food_index, obj_index.

        # Check if the character is not holding any food (nothing to put inside an object). If true, return a punishment reward.
        if self.observation['food_holding'] <= 0:
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # Check if the target food exists in the environment. If it doesn't exist, return a punishment.
        if not self.observation['food_state'][food_index] & FoodStateBitmapEnum.EXIST:
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # Check if the target object exists in the environment. If it doesn't exist, return a punishment.
        if not self.observation['object_state'][obj_index] & ObjectStateBitmapEnum.EXIST:
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # Check if the object can be opened and is open. If it can't be opened or isn't open, return a punishment.
        if not (self.observation['object_state'][obj_index] & ObjectStateBitmapEnum.CAN_OPEN and
                self.observation['object_state'][obj_index] & ObjectStateBitmapEnum.OPEN):
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # Check if the character is holding the target food. If not, return a punishment.
        if not self.observation['food_character_relation'][food_index] == FoodCharacterStateEnum.HOLD:
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # Check if the character is close to the target object (e.g., fridge). If not, return a punishment.
        if not self.observation['object_character_relation'][obj_index] == ObjectCharacterStateEnum.CLOSE_TO:
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # If all checks pass, the character is putting the food inside the object (e.g., inside the fridge).

        # Update the food-character relation: change from 'HOLD' to 'CLOSE_TO' for the target food.
        self.observation['food_character_relation'][food_index] = FoodCharacterStateEnum.CLOSE_TO | 0

        # Update the food-object relation: set the relation to 'INSIDE' for the target food and object (e.g., fridge).
        self.observation['food_object_relation'][food_index][obj_index] = FoodObjectStateEnum.INSIDE | 0

        self.observation['food_holding'] -= 1

        # Process and append the virtual-home instruction for this action to the instruction list.
        instruction = self._process_step_instruction(action)
        self.vh_metadata['instruction_list'].append(instruction)

        agent_reward = self._process_food_in_fridge()
        # Notice: We don't check here because we do this at the beginning of the function `self.reset`
        # if self.observation['food_in_fridge'] >= self.vh_metadata['food_exist_count']:
        #     return self.observation, agent_reward, True, False, self.vh_metadata

        # Return the updated observation, the calculated reward, and the metadata
        return self.observation, agent_reward, is_done, False, self.vh_metadata

    def _action_open(
            self,
            action: ActType,
            is_done: bool,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Executes the action of opening an object in the environment (e.g., opening a fridge, cupboard).

        Args:
            action (ActType): A tuple containing the action type, food index, and object index.
                - action_type (ActionEnum): The type of action to execute (in this case, opening an object).
                - food_index (int): The index of the food item (not used in this specific action but included for consistency).
                - obj_index (int): The index of the object to be opened (e.g., fridge, cupboard).

            is_done (bool): A boolean indicating whether the episode has already ended.

        Returns:
            tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
                A tuple containing:
                - The updated observation after the action is executed.
                - A reward (0 in this case, as the action is not inherently rewarded or punished).
                - A boolean indicating whether the episode has ended.
                - A boolean indicating whether the episode was truncated.
                - A dictionary containing metadata (`vh_metadata`), including the list of instructions.

        Explanation:
            This function handles the action of opening an object in the environment, such as opening a fridge or cupboard.
            - It checks whether the object exists and if it can be opened. If the object is not openable or does not exist, it returns a punishment.
            - It verifies that the character is close enough to the target object to perform the action.
            - If all conditions are met, the object is opened by updating its state to reflect that it's open.
            - The function then appends the action instruction to the virtual-home instruction list and returns the updated state.
        """

        _, food_index, obj_index = action  # Unpack the action tuple: action_type, food_index, obj_index.

        # Check if the object exists in the environment. If it doesn't exist, return a punishment.
        if not self.observation['object_state'][obj_index] & ObjectStateBitmapEnum.EXIST:
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # Check if the object can be opened. If it can't be opened, return a punishment.
        if not self.observation['object_state'][obj_index] & ObjectStateBitmapEnum.CAN_OPEN:
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # Check if the object is already open. If it is, return a punishment.
        if self.observation['object_state'][obj_index] & ObjectStateBitmapEnum.OPEN:
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # Check if the character is close to the target object. If not, return a punishment.
        if not self.observation['object_character_relation'][obj_index] == ObjectCharacterStateEnum.CLOSE_TO:
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # Check if the character has no hand. If not, return a punishment
        if self.observation['food_holding'] >= 2:
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # Open the object by updating its state to reflect that it's now open.
        self.observation['object_state'][obj_index] |= ObjectStateBitmapEnum.OPEN

        # Process and append the virtual-home instruction for this action to the instruction list.
        instruction = self._process_step_instruction(action)
        self.vh_metadata['instruction_list'].append(instruction)

        reward = 0
        if self.OBJECT_LIST[obj_index] == 'fridge' \
                and self.observation['food_holding'] > 0:
            reward += self.TASK_GUIDE_REWARD

        # Return the updated observation, no reward for this action (0), and the metadata.
        return self.observation, reward, is_done, False, self.vh_metadata

    def _action_close(
            self,
            action: ActType,
            is_done: bool,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Executes the action of closing an object in the environment (e.g., closing a fridge, cupboard).

        Args:
            action (ActType): A tuple containing the action type, food index, and object index.
                - action_type (ActionEnum): The type of action to execute (in this case, closing an object).
                - food_index (int): The index of the food item (not used in this specific action but included for consistency).
                - obj_index (int): The index of the object to be closed (e.g., fridge, cupboard).

            is_done (bool): A boolean indicating whether the episode has already ended.

        Returns:
            tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
                A tuple containing:
                - The updated observation after the action is executed.
                - A reward (0 in this case, as the action is not inherently rewarded or punished).
                - A boolean indicating whether the episode has ended.
                - A boolean indicating whether the episode was truncated.
                - A dictionary containing metadata (`vh_metadata`), including the list of instructions.

        Explanation:
            This function handles the action of closing an object in the environment, such as closing a fridge or cupboard.
            - It checks whether the object exists and if it can be opened. If the object cannot be closed or does not exist, it returns a punishment.
            - It verifies that the object is currently open and that the character is close enough to the target object to perform the action.
            - If all conditions are met, the object is closed by updating its state to reflect that it is no longer open.
            - The function then appends the action instruction to the virtual-home instruction list and returns the updated state.
        """

        _, food_index, obj_index = action  # Unpack the action tuple: action_type, food_index, obj_index.

        # Check if the object exists in the environment. If it doesn't exist, return a punishment.
        if not self.observation['object_state'][obj_index] & ObjectStateBitmapEnum.EXIST:
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # Check if the object can be opened (and closed). If it can't be opened or closed, return a punishment.
        if not self.observation['object_state'][obj_index] & ObjectStateBitmapEnum.CAN_OPEN:
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # Check if the object is already closed. If it's not open, return a punishment.
        if not self.observation['object_state'][obj_index] & ObjectStateBitmapEnum.OPEN:
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # Check if the character is close enough to the target object to perform the action.
        if not self.observation['object_character_relation'][obj_index] == ObjectCharacterStateEnum.CLOSE_TO:
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # Check if the character has no hand. If not, return a punishment
        if self.observation['food_holding'] >= 2:
            return self.observation, self.PUNISHMENT_REWARD, is_done, False, self.vh_metadata

        # Close the object by updating its state to reflect that it's now closed.
        self.observation['object_state'][obj_index] &= ~ObjectStateBitmapEnum.OPEN

        # Process and append the virtual-home instruction for this action to the instruction list.
        instruction = self._process_step_instruction(action)
        self.vh_metadata['instruction_list'].append(instruction)

        reward = 0
        if self.OBJECT_LIST[obj_index] == 'fridge' \
                and self.observation['food_holding'] == 0:
            reward += self.TASK_GUIDE_REWARD

        # Return the updated observation, no reward for this action (0), and the metadata.
        return self.observation, reward, is_done, False, self.vh_metadata

    def _process_step_instruction(self, action: ActType) -> str:
        self.none = None
        action_type, food_index, obj_index = action

        if action_type == ActionEnum.WALK_TO_FOOD:
            food_class_name = self.FOOD_LIST[food_index]
            food_vh_id = self.vh_metadata['food_id_bidict'][food_class_name]
            return f'<char0> [walk] <{food_class_name}> ({food_vh_id})'
        elif action_type == ActionEnum.WALK_TO_OBJECT:
            object_class_name = self.OBJECT_LIST[obj_index]
            object_vh_id = self.vh_metadata['object_id_bidict'][object_class_name]
            return f'<char0> [walk] <{object_class_name}> ({object_vh_id})'
        elif action_type == ActionEnum.GRAB:
            food_class_name = self.FOOD_LIST[food_index]
            food_vh_id = self.vh_metadata['food_id_bidict'][food_class_name]
            return f'<char0> [grab] <{food_class_name}> ({food_vh_id})'
        elif action_type == ActionEnum.PUT:
            food_class_name = self.FOOD_LIST[food_index]
            food_vh_id = self.vh_metadata['food_id_bidict'][food_class_name]
            object_class_name = self.OBJECT_LIST[obj_index]
            object_vh_id = self.vh_metadata['object_id_bidict'][object_class_name]
            return f'<char0> [put] <{food_class_name}> ({food_vh_id}) <{object_class_name}> ({object_vh_id})'
        elif action_type == ActionEnum.PUTIN:
            food_class_name = self.FOOD_LIST[food_index]
            food_vh_id = self.vh_metadata['food_id_bidict'][food_class_name]
            object_class_name = self.OBJECT_LIST[obj_index]
            object_vh_id = self.vh_metadata['object_id_bidict'][object_class_name]
            return f'<char0> [putin] <{food_class_name}> ({food_vh_id}) <{object_class_name}> ({object_vh_id})'
        elif action_type == ActionEnum.OPEN:
            object_class_name = self.OBJECT_LIST[obj_index]
            object_vh_id = self.vh_metadata['object_id_bidict'][object_class_name]
            return f'<char0> [open] <{object_class_name}> ({object_vh_id})'
        elif action_type == ActionEnum.CLOSE:
            object_class_name = self.OBJECT_LIST[obj_index]
            object_vh_id = self.vh_metadata['object_id_bidict'][object_class_name]
            return f'<char0> [close] <{object_class_name}> ({object_vh_id})'
        return ''

    def _process_environment_graph(self, g: Dict[str, Any]):
        """
        Processes the environment graph to update the observation and metadata.

        Args:
            g (g: Dict[str, Any]): The environment graph returned by UnityCommunication.
        """
        action_count, food_count, object_count = self.get_action_count(), self.get_food_count(), self.get_object_count()
        food_index_dict, object_index_dict = self.get_food_index_dict(), self.get_object_index_dict()
        # 1. Find all objects in the environment graph
        observation_space_dtype = self.OBSERVATION_SPACE_DTYPE
        food_state_list = np.array(
            [FoodStateBitmapEnum.NONE | 0 for _ in range(food_count)],
            dtype=observation_space_dtype,
        )
        obj_state_list = np.array(
            [ObjectStateBitmapEnum.NONE | 0 for _ in range(object_count)],
            dtype=observation_space_dtype,
        )

        for node in g['nodes']:
            if node['class_name'] in self.FOOD_LIST:
                target_food_index = food_index_dict[node['class_name']]
                food_state_list[target_food_index] |= FoodStateBitmapEnum.EXIST
                self.vh_metadata['food_id_bidict'][node['class_name']] = node['id']

            if node['class_name'] in self.OBJECT_LIST:
                target_object_index = object_index_dict[node['class_name']]
                obj_state_list[target_object_index] |= ObjectStateBitmapEnum.EXIST

                if 'CAN_OPEN' in node['properties']:
                    obj_state_list[target_object_index] |= ObjectStateBitmapEnum.CAN_OPEN
                    if 'OPEN' in node['states']:
                        obj_state_list[target_object_index] |= ObjectStateBitmapEnum.OPEN

                if 'HAS_SWITCH' in node['properties']:
                    obj_state_list[target_object_index] |= ObjectStateBitmapEnum.HAS_SWITCH
                    if 'ON' in node['states']:
                        obj_state_list[target_object_index] |= ObjectStateBitmapEnum.TURNED_ON

                self.vh_metadata['object_id_bidict'][node['class_name']] = node['id']

            if node['class_name'] == 'character':
                self.vh_metadata['character_id'] = node['id']

        # 2. Find all relations (edges) in the environment graph
        food_character_relation = np.array(
            [FoodCharacterStateEnum.NONE | 0 for _ in range(food_count)],
            dtype=observation_space_dtype,
        )
        object_character_relation = np.array(
            [ObjectCharacterStateEnum.NONE | 0 for _ in range(object_count)],
            dtype=observation_space_dtype,
        )
        food_object_relation = np.array(
            [[FoodObjectStateEnum.NONE | 0 for _ in range(object_count)] for _ in range(food_count)],
            dtype=observation_space_dtype,
        )
        food_holding = 0

        for edge in g['edges']:
            if edge['from_id'] in self.vh_metadata['food_id_bidict'].inverse \
                    and edge['to_id'] in self.vh_metadata['object_id_bidict'].inverse:

                food_class_name = self.vh_metadata['food_id_bidict'].inverse[edge['from_id']]
                food_index = food_index_dict[food_class_name]

                object_class_name = self.vh_metadata['object_id_bidict'].inverse[edge['to_id']]
                object_index = object_index_dict[object_class_name]

                if edge['relation_type'] == 'ON':
                    food_object_relation[food_index][object_index] = FoodObjectStateEnum.ON | 0
                elif edge['relation_type'] == 'INSIDE':
                    food_object_relation[food_index][object_index] = FoodObjectStateEnum.INSIDE | 0
                elif edge['relation_type'] == 'FACING':
                    food_object_relation[food_index][object_index] = FoodObjectStateEnum.FACING | 0

            if edge['from_id'] == self.vh_metadata['character_id'] and \
                    edge['to_id'] in self.vh_metadata['object_id_bidict'].inverse:

                object_class_name = self.vh_metadata['object_id_bidict'].inverse[edge['to_id']]
                object_index = object_index_dict[object_class_name]

                if edge['relation_type'] == 'CLOSE':
                    object_character_relation[object_index] = ObjectCharacterStateEnum.CLOSE_TO | 0
                elif edge['relation_type'] == 'FACING':
                    object_character_relation[object_index] = ObjectCharacterStateEnum.FACING | 0

            if edge['from_id'] == self.vh_metadata['character_id'] and \
                    edge['to_id'] in self.vh_metadata['food_id_bidict'].inverse:

                food_class_name = self.vh_metadata['food_id_bidict'].inverse[edge['to_id']]
                food_index = food_index_dict[food_class_name]

                if edge['relation_type'] == 'CLOSE':
                    food_character_relation[food_index] = FoodCharacterStateEnum.CLOSE_TO | 0
                elif edge['relation_type'] == 'HOLD_RH' or edge['relation_type'] == 'HOLD_LH':
                    food_character_relation[food_index] = FoodCharacterStateEnum.HOLD | 0
                    food_holding += 1

        food_in_fridge = 0
        fridge_object_index = object_index_dict['fridge']
        if obj_state_list[fridge_object_index] & ObjectStateBitmapEnum.EXIST:
            for i in range(len(food_object_relation)):
                if food_object_relation[i][fridge_object_index] & FoodObjectStateEnum.INSIDE:
                    food_in_fridge += 1
            self.vh_metadata['fridge_exist_flag'] = True
        else:
            self.vh_metadata['fridge_exist_flag'] = False

        # 3. Update the observation with the processed data
        self.observation = {
            'food_state': food_state_list,
            'object_state': obj_state_list,
            'food_character_relation': food_character_relation,
            'object_character_relation': object_character_relation,
            'food_object_relation': food_object_relation,
            'food_holding': food_holding,
            'food_in_fridge': food_in_fridge,
        }

        food_exist_count = 0
        for i in range(len(food_state_list)):
            if food_state_list[i] & FoodStateBitmapEnum.EXIST:
                food_exist_count += 1

        self.vh_metadata['food_exist_count'] = food_exist_count

    def _process_reset_metadata(self) -> Dict[str, Any]:
        """
        Resets the metadata to its initial state.
        """
        self.none = None
        return {
            'character_id': 0,  # ID of the character in the environment
            'food_id_bidict': bidict({}),  # Bidirectional mapping between food names and IDs
            'object_id_bidict': bidict({}),  # Bidirectional mapping between object names and IDs
            'step': 0,  # Current step count in the episode
            'instruction_list': [],  # Instruction list to execute by virtual home
            'food_exist_count': 0,  # All the existing food in the environment
            'fridge_exist_flag': False  # If the fridge exist
        }

    def _process_food_in_fridge(self) -> int:
        """
            Process the food items in the fridge and calculate the reward based on the number of food items inside.

            Returns:
                int: The calculated agent reward based on the number of food items inside the fridge.

            Explanation:
                This method counts how many food items are currently inside the fridge by checking the
                food-object relation. The reward is calculated by comparing the current count of food in the
                fridge to the target number of food items and applying the reward based on the difference.
                The reward is then returned and the current food count in the fridge is updated in the observation.

                The reward calculation is as follows:
                - If the food count inside the fridge is higher than the target, the agent receives a positive reward.
                - If the food count inside the fridge is lower than the target, the agent receives a negative reward.

            """
        action_count, food_count, object_count = self.get_action_count(), self.get_food_count(), self.get_object_count()
        object_index_dict = self.get_object_index_dict()
        # Reward Calculation: Check if the food was successfully put into the fridge
        food_in_fridge = 0
        object_fridge_index = object_index_dict['fridge']

        # Count the number of food items inside the fridge
        for i in range(food_count):
            if self.observation['food_object_relation'][i][object_fridge_index] == FoodObjectStateEnum.INSIDE:
                food_in_fridge += 1

        # The reward is based on the number of food items inside the fridge compared to the target food count
        agent_reward = (food_in_fridge - self.observation['food_in_fridge']) * self.TARGET_REACHED_REWARD
        self.observation['food_in_fridge'] = food_in_fridge
        return agent_reward

# class VirtualHomeEnv(gym.Env):
#     def __init__.py(self) -> None:
#         super(VirtualHomeEnv, self).__init__.py()
#
#     def reset(
#             self,
#             *,
#             seed: int | None = None,
#             options: dict[str, Any] | None = None,
#     ) -> tuple[ObsType, dict[str, Any]]:
#         pass
#
#     def step(
#             self, action: ActType
#     ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
#         pass
#
#     def render(self):
#         pass
#
#     def close(self):
#         pass
