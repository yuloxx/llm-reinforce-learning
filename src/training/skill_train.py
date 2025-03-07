from .trainer import ModelTrainer, ModelTrainerRLAlgo
from src.vh_util.create_env import get_basic_environment_graph
from src.vh_env.cross_env import CrossVirtualHomeGatherFoodEnvV2

YOUR_FILE_NAME = "D:\\programs\\windows_exec.v2.2.4\\VirtualHome.exe"

def train_cross_food_gather_agent_v1() -> ModelTrainer:

    env_graph_list = []
    for i in range(4):
        g = get_basic_environment_graph(virtualhome_exec_path=YOUR_FILE_NAME)
        env_graph_list.append(g)

    trainer = ModelTrainer(
        algo=ModelTrainerRLAlgo.PPO,
        create_env=lambda: CrossVirtualHomeGatherFoodEnvV2(env_graph_list),
    )

    hyperparameter = 'vf_coef'
    hyperparameter_list = [{hyperparameter: i} for i in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]]
    trainer.train(
        vec_envs=4,
        total_timesteps=1200000,
        hyperparameters_list=hyperparameter_list
    )

    trainer.show_mean_reward(hyperparameter)
    trainer.show_value_loss(hyperparameter)
    trainer.show_entropy_loss(hyperparameter)

    return trainer


