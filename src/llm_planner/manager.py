import json
import time
from stable_baselines3 import PPO
from virtualhome.simulation.unity_simulator import UnityCommunication
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from typing import TypeVar, Any
from langchain.tools import tool
from ..vh_env import VirtualHomeGatherFoodEnvV2


class VhManager:
    """
    A manager that integrates VirtualHome with a language model to assist in planning tasks.
    It allows listing available food items, selecting characters, and generating action lists
    using a pre-trained reinforcement learning model.
    """
    FOOD_LIST = [
        'salmon', 'pie', 'mincedmeat', 'juice', 'pancake', 'pear',
        'milkshake', 'salad', 'wine', 'chocolatesyrup', 'chicken', 'carrot'
    ]

    VH_ENV_COUNT = 7

    CHARACTER_LIST = [
        'Chars/Male1', 'Chars/Female1', 'Chars/Male2', 'Chars/Female2',
        'Chars/Male6', 'Chars/Female4',
    ]
    SYSTEM_PROMPT = (
        "You are a task planning assistant for a VirtualHome simulation environment. "
        "Your goal is to help users generate action plans for characters to collect food. "
        "Use the available functions to retrieve environment information and generate action lists."
    )

    def __init__(self):
        self.comm = None
        self.model_path = None
        self.env_list = []
        self.model = None
        self.agent = None

    def initialize(self, comm: UnityCommunication, chat_llm: ChatOpenAI, model_path: str):
        """
        Initializes the VirtualHome manager with a Unity communication module, a language model,
        and a pre-trained reinforcement learning model.

        Args:
            comm (UnityCommunication): Communication module with the VirtualHome environment.
            chat_llm (ChatOpenAI): Language model for planning and reasoning.
            model_path (str): Path to the pre-trained RL model for generating actions.
        """
        self.comm = comm
        self.env_list = []
        self.model = PPO.load(model_path)
        self.agent = initialize_agent(
            tools=tools,
            llm=chat_llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

        self._init_env_graph()

    def _init_env_graph(self):

        for i in range(VhManager.VH_ENV_COUNT):
            self.comm.reset(i)
            res, g = self.comm.environment_graph()
            self.env_list.append(g)
            time.sleep(1)

    def serve(self, command: str):
        """
        Processes a command using the language model agent to invoke the appropriate functions.

        Args:
            command (str): The user-provided command for planning actions.
        """
        response = self.agent.invoke({
            "input": command,
            "system_input": VhManager.SYSTEM_PROMPT
        })
        print(json.dumps(response, indent=2, ensure_ascii=False))

manager_instance = VhManager()


PromptType = TypeVar('PromptType', bound=str)


@tool
def list_food_in_env() -> PromptType:
    """
    Retrieves a list of food items available in each VirtualHome environment.

    Returns:
        str: A dict containing an empty prompt string and a dictionary where
               keys are environment IDs and values are lists of available food items.
    """

    prompt = "Good! Please check the dictionary and find one proper environment ID which contains all food items which user are interested in."
    res = {}
    for i in range(VhManager.VH_ENV_COUNT):
        res[i] = []
        for node in manager_instance.env_list[i]['nodes']:
            if node['class_name'] in VhManager.FOOD_LIST:
                res[i].append(node['class_name'])

    return json.dumps({
        'prompt': prompt,
        'result': res,
    })


@tool
def list_supported_character() -> PromptType:
    """
    Lists all supported character models that can be used in the environment.

    Returns:
        str: A dict containing a system prompt and a list of supported character models.
    """

    prompt = "This is the supported character, please choose a proper one in `generate_action_list` task"
    return json.dumps({
        'prompt': prompt,
        'result': VhManager.CHARACTER_LIST,
    })


@tool
def generate_action_list(character_name: str, env_id: int) -> PromptType:
    """
    Generates an action list for a specified character in a given environment using a trained RL model.

    Args:
        character_name (str): The name of the character to control.
        env_id (int): The ID of the environment in which the character operates.

    Returns:
        str: A dict containing a success or error message and either the generated action list or None.
    """
    prompt_character_name_error = "Sorry, provided character name is not in pre-defined character list."
    prompt_success = "Congratulations! Action list has been successfully generated!"

    if character_name not in VhManager.CHARACTER_LIST:
        return json.dumps({
        'prompt': prompt_character_name_error,
    })
    manager_instance.comm.reset(env_id)
    manager_instance.comm.add_character(character_name)
    _, g = manager_instance.comm.environment_graph()

    done = False

    env = VirtualHomeGatherFoodEnvV2(environment_graph=manager_instance.env_list[env_id])
    obs, info = env.reset()

    while not done:
        action, _ = manager_instance.model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
    action_list = env.get_instruction_list()
    return json.dumps({
        'prompt': prompt_success,
        'result': action_list,
    })

tools = [
    list_food_in_env,
    list_supported_character,
    generate_action_list,
]
