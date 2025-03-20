import os
import unittest
from pathlib import Path

from src.vh_util.create_env import get_basic_environment_graph
from stable_baselines3.ppo import PPO
from src.vh_env.cross_env import CrossVirtualHomeGatherFoodEnvV2
from src.training.evaluator import ModelEvaluator

class EvaluatorTest(unittest.TestCase):

    def setUp(self):
        # pass
        YOUR_FILE_NAME = "D:\\programs\\windows_exec.v2.2.4\\VirtualHome.exe"
        self.env_graph_list = []
        for i in range(4):
            g = get_basic_environment_graph(virtualhome_exec_path=YOUR_FILE_NAME, environment_index=i)
            self.env_graph_list.append(g)

    def tearDown(self):
        pass

    def test_model_evaluator(self):
        models = {}
        env = CrossVirtualHomeGatherFoodEnvV2(self.env_graph_list)
        for h in [0.1, 0.3, 0.5, 0.7, 0.9]:
            name  = f'vfcoef{h}'
            model_t = PPO.load(Path(f'../model/cross_food_gather_agent_vfcoef{h}'))
            models[name] = model_t

        model_evaluator = ModelEvaluator(
            models = models,
            env = env,
        )

        model_evaluator.evaluate(
            episode = 4,
            max_steps = 64,
        )

        model_evaluator.show_mean_reward()
        model_evaluator.show_mean_episode_length()

    def test_basic_eval(self):
        from src.training.skill_train import train_cross_food_gather_agent_v2

        train_cross_food_gather_agent_v2()


    def test_model(self):
        from src.vh_env.food_gather_env import VirtualHomeGatherFoodEnvV2
        from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
        YOUR_FILE_NAME = "D:\\programs\\windows_exec.v2.2.4\\VirtualHome.exe"
        comm = UnityCommunication(file_name=YOUR_FILE_NAME)
        comm.reset(0)
        comm.add_character()
        res, g = comm.environment_graph()
        comm.close()
        # print(g)

        model = PPO.load("../model/v2_first_test.zip")
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


