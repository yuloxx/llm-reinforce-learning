import pprint
import unittest

from langchain_openai import ChatOpenAI

from virtualhome.simulation.unity_simulator import UnityCommunication
from src.llm_planner.manager import VhManager
from src.llm_planner.manager import manager_instance

class TestLLMManager(unittest.TestCase):

    def setUp(self):
        YOUR_FILE_NAME = "D:\\programs\\windows_exec.v2.2.4\\VirtualHome.exe"
        comm = UnityCommunication(file_name=YOUR_FILE_NAME)

        chat_llm = ChatOpenAI(
            model='deepseek-ai/DeepSeek-V3',
            openai_api_key='sk-ijvudamukvchschgkbchifkitkcmsvoyfvlfnaemznbdivwq',
            base_url='https://api.siliconflow.cn/v1/'
        )

        manager_instance.initialize(comm=comm, chat_llm=chat_llm, model_path='../model/cross_food_gather_agent_vfcoef0.1')

    def test_tool_list_food_in_env(self):
        # print(self.manger.list_food_in_env())
        pass

    def test_tool_list_supported_character(self):
        # print(self.manger.list_supported_character())
        pass

    def test_tool_generate_action(self):
        # res = self.manger.generate_action_list(character_name='Chars/Male1', env_id=2, model_path="../model/cross_food_gather_agent_vfcoef0.1")
        # pprint.pprint(res)
        pass

    def test_agent(self):


        manager_instance.serve(command="I want to see how agent named Chars/Male1 act in an environment with salmon")


