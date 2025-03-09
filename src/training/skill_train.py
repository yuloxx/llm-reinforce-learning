from .trainer import ModelTrainer, ModelTrainerRLAlgo
from src.vh_util.create_env import get_basic_environment_graph
from src.vh_env.cross_env import CrossVirtualHomeGatherFoodEnvV2

YOUR_FILE_NAME = "D:\\programs\\windows_exec.v2.2.4\\VirtualHome.exe"

def train_cross_food_gather_agent_v1() -> ModelTrainer:

    env_graph_list = []
    for i in range(4):
        g = get_basic_environment_graph(
            virtualhome_exec_path=YOUR_FILE_NAME,
            environment_index=i
        )
        env_graph_list.append(g)

    trainer = ModelTrainer(
        algo=ModelTrainerRLAlgo.PPO,
        create_env=lambda: CrossVirtualHomeGatherFoodEnvV2(env_graph_list),
    )

    hyperparameter = 'vf_coef'
    hyperparameter_list = [{hyperparameter: i} for i in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]]
    trainer.train(
        vec_envs=4,
        total_timesteps=640_000,
        hyperparameters_list=hyperparameter_list
    )

    trainer.show_mean_reward(hyperparameter)
    trainer.show_value_loss(hyperparameter)
    trainer.show_entropy_loss(hyperparameter)

    return trainer


def train_cross_food_gather_agent_v2() -> ModelTrainer:

    env_graph_list = []
    for i in range(4):
        g = get_basic_environment_graph(
            virtualhome_exec_path=YOUR_FILE_NAME,
            environment_index=i
        )
        env_graph_list.append(g)

    trainer = ModelTrainer(
        algo=ModelTrainerRLAlgo.PPO,
        create_env=lambda: CrossVirtualHomeGatherFoodEnvV2(env_graph_list),
    )

    hyperparameter = 'vf_coef'
    hyperparameter_list = [{hyperparameter: i} for i in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]]
    res = trainer.train(
        vec_envs=4,
        total_timesteps=160000,
        hyperparameters_list=hyperparameter_list
    )

    # trainer.show_mean_reward(hyperparameter)
    # trainer.show_value_loss(hyperparameter)
    # trainer.show_entropy_loss(hyperparameter)

    env = CrossVirtualHomeGatherFoodEnvV2(env_graph_list)
    models = {f'model_{i}': model_x['model'] for i, model_x in enumerate(res)}

    from src.training.evaluator import ModelEvaluator
    model_evaluator = ModelEvaluator(
        models=models,
        env=env,
    )

    model_evaluator.evaluate(
        episode=4,
        max_steps=64,
    )

    model_evaluator.show_mean_reward()
    model_evaluator.show_mean_episode_length()

    model_x = res[0]['model']

    obs, _ = env.reset()
    reward = 0
    for _ in range(64):
        action, _ = model_x.predict(obs, deterministic=True)
        obs, reward_t, done,_ , _ = env.step(action)
        reward += reward_t

    print(reward)
    print(env.get_instruction_list())

    return trainer
