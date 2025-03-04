from src.training import ModelTrainer, ModelTrainerRLAlgo
import unittest
from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
from src.vh_env import VirtualHomeGatherFoodEnvV2


class ModelTrainerTest(unittest.TestCase):
    YOUR_FILE_NAME = "D:\\programs\\windows_exec.v2.2.4\\VirtualHome.exe"
    def setUp(self):

        self.comm = UnityCommunication(file_name=self.YOUR_FILE_NAME)

    def tearDown(self):
        self.comm.close()


    def test_model_trainer(self):
        self.comm.reset(0)
        self.comm.add_character()
        res, g = self.comm.environment_graph()

        model_trainer = ModelTrainer(
            algo=ModelTrainerRLAlgo.PPO,
            create_env= lambda: VirtualHomeGatherFoodEnvV2(environment_graph=g)
        )

        model_trainer.train(
            vec_envs=4,
            total_timesteps=12000,
            hyperparameters_list=[{}],
        )

    def test_model_trainer_show_mean_reward(self):
        self.comm.reset(0)
        self.comm.add_character()

        model_trainer = ModelTrainer(
            algo=ModelTrainerRLAlgo.PPO,
            create_env= lambda: VirtualHomeGatherFoodEnvV2(environment_graph=g)
        )

        model_trainer.show_mean_reward_chart('bbb')

    def test_model_trainer_show_episode_length(self):
        self.comm.reset(0)
        self.comm.add_character()
        model_trainer = ModelTrainer(
            algo=ModelTrainerRLAlgo.PPO,
            create_env= lambda: VirtualHomeGatherFoodEnvV2(environment_graph=g)
        )

        model_trainer.show_ep_length_chart('aaa')

    def test_model_trainer_show_success_rate(self):
        self.comm.reset(0)
        self.comm.add_character()
        model_trainer = ModelTrainer(
            algo=ModelTrainerRLAlgo.PPO,
            create_env= lambda: VirtualHomeGatherFoodEnvV2(environment_graph=g)
        )

        model_trainer.show_success_rate_chart('aaa')
