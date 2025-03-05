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
            algo=ModelTrainerRLAlgo.A2C,
            create_env= lambda: VirtualHomeGatherFoodEnvV2(environment_graph=g)
        )

        model_trainer.train(
            vec_envs=4,
            total_timesteps=120000,
            hyperparameters_list=[{}],
        )

    def test_compare_show_final_reward(self):
        self.comm.reset(0)
        self.comm.add_character()
        _, g = self.comm.environment_graph()

        model_trainer = ModelTrainer(
            algo=ModelTrainerRLAlgo.PPO,
            create_env= lambda: VirtualHomeGatherFoodEnvV2(environment_graph=g)
        )

        model_trainer.compare_show_final_mean_reward('bbb')

    def test_compare_show_final_episode_length(self):
        self.comm.reset(0)
        self.comm.add_character()
        _, g = self.comm.environment_graph()

        model_trainer = ModelTrainer(
            algo=ModelTrainerRLAlgo.PPO,
            create_env= lambda: VirtualHomeGatherFoodEnvV2(environment_graph=g)
        )

        model_trainer.compare_show_final_ep_length('aaa')

    def test_compare_show_final_success_rate(self):
        self.comm.reset(0)
        self.comm.add_character()
        _, g = self.comm.environment_graph()

        model_trainer = ModelTrainer(
            algo=ModelTrainerRLAlgo.PPO,
            create_env= lambda: VirtualHomeGatherFoodEnvV2(environment_graph=g)
        )

        model_trainer.compare_show_final_success_rate('aaa')


    def test_train_hyperparameters(self):
        self.comm.reset(0)
        self.comm.add_character()

        # for gamma in range(90, 99):
        #     print(gamma)

        _, g = self.comm.environment_graph()

        model_trainer = ModelTrainer(
            algo=ModelTrainerRLAlgo.PPO,
            create_env= lambda: VirtualHomeGatherFoodEnvV2(environment_graph=g)
        )
        hyperparameter_name = 'vf_coef'
        # hyperparameters_list = [{hyperparameter_name: n_steps} for n_steps in [256, 512, 1024, 2048]]
        hyperparameters_list = [{hyperparameter_name: float(vf_coef) / 10} for vf_coef in range(1, 10)]
        print(hyperparameters_list)
        model_trainer.train(
            vec_envs=2,
            total_timesteps=80000,
            hyperparameters_list=hyperparameters_list,
        )
        print(model_trainer.result_list)
        model_trainer.compare_show_final_mean_reward(hyperparameter_name)


    def test_show_learning_rate(self):
        self.comm.reset(0)
        self.comm.add_character()

        # for gamma in range(90, 99):
        #     print(gamma)

        _, g = self.comm.environment_graph()

        model_trainer = ModelTrainer(
            algo=ModelTrainerRLAlgo.PPO,
            create_env= lambda: VirtualHomeGatherFoodEnvV2(environment_graph=g)
        )
        hyperparameter_name = 'n_steps'
        hyperparameters_list = [{hyperparameter_name: n_steps} for n_steps in [256, 512, 1024, 2048]]
        # hyperparameters_list = [{hyperparameter_name: float(vf_coef) / 10} for vf_coef in range(1, 10)]
        # print(hyperparameters_list)
        model_trainer.train(
            vec_envs=2,
            total_timesteps=80000,
            hyperparameters_list=hyperparameters_list,
        )
        print(model_trainer.result_list)
        model_trainer.show_learning_rate(hyperparameter_name)

    def test_show_entropy_loss(self):
        self.comm.reset(0)
        self.comm.add_character()

        # for gamma in range(90, 99):
        #     print(gamma)

        _, g = self.comm.environment_graph()

        model_trainer = ModelTrainer(
            algo=ModelTrainerRLAlgo.PPO,
            create_env= lambda: VirtualHomeGatherFoodEnvV2(environment_graph=g)
        )
        hyperparameter_name = 'n_steps'
        hyperparameters_list = [{hyperparameter_name: n_steps} for n_steps in [256, 512, 1024, 2048]]
        # hyperparameters_list = [{hyperparameter_name: float(vf_coef) / 10} for vf_coef in range(1, 10)]
        # print(hyperparameters_list)
        model_trainer.train(
            vec_envs=2,
            total_timesteps=80000,
            hyperparameters_list=hyperparameters_list,
        )
        # print(model_trainer.result_list)
        model_trainer.show_entropy_loss(hyperparameter_name)

    def test_show_value_loss(self):
        self.comm.reset(0)
        self.comm.add_character()

        # for gam12  ma in range(90, 99):
        #     print(gamma)

        _, g = self.comm.environment_graph()

        model_trainer = ModelTrainer(
            algo=ModelTrainerRLAlgo.PPO,
            create_env= lambda: VirtualHomeGatherFoodEnvV2(environment_graph=g)
        )
        hyperparameter_name = 'n_steps'
        hyperparameters_list = [{hyperparameter_name: n_steps} for n_steps in [2048]]
        # hyperparameters_list = [{hyperparameter_name: float(vf_coef) / 10} for vf_coef in range(1, 10)]
        # print(hyperparameters_list)
        model_trainer.train(
            vec_envs=2,
            total_timesteps=1000000,
            hyperparameters_list=hyperparameters_list,
        )
        # print(model_trainer.result_list)
        model_trainer.show_value_loss(hyperparameter_name)

    def test_show_mean_reward(self):
        self.comm.reset(0)
        self.comm.add_character()
        _, g = self.comm.environment_graph()

        model_trainer = ModelTrainer(
            algo=ModelTrainerRLAlgo.PPO,
            create_env= lambda: VirtualHomeGatherFoodEnvV2(environment_graph=g)
        )
        hyperparameter_name = 'n_steps'
        hyperparameters_list = [{hyperparameter_name: n_steps} for n_steps in [2048]]
        model_trainer.train(
            vec_envs=2,
            total_timesteps=80000,
            hyperparameters_list=hyperparameters_list,
        )
        model_trainer.show_mean_reward(hyperparameter_name)

    def test_show_episode_length(self):
        self.comm.reset(0)
        self.comm.add_character()
        _, g = self.comm.environment_graph()

        model_trainer = ModelTrainer(
            algo=ModelTrainerRLAlgo.PPO,
            create_env=lambda: VirtualHomeGatherFoodEnvV2(environment_graph=g)
        )
        hyperparameter_name = 'n_steps'
        hyperparameters_list = [{hyperparameter_name: n_steps} for n_steps in [2048]]
        model_trainer.train(
            vec_envs=2,
            total_timesteps=80000,
            hyperparameters_list=hyperparameters_list,
        )
        model_trainer.show_ep_length(hyperparameter_name)




