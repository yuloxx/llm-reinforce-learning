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
            total_timesteps=10000,
            hyperparameters_list=[{}],
        )
