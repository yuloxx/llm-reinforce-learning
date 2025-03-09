import torch
from stable_baselines3.common.policies import ActorCriticPolicy


class MyACModel(torch.nn.Module):
    def __init__(
        self,
        features_dim,
        last_layer_dim_pi=64,
        last_layer_dim_vf=64
    ):
        super(MyACModel, self).__init__()

        # Store the output dimensions for the Actor and Critic
        self.latent_dim_pi = last_layer_dim_pi  # Output dimension of the policy network (Actor)
        self.latent_dim_vf = last_layer_dim_vf  # Output dimension of the value function network (Critic)

        # Define the Actor network
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(features_dim, last_layer_dim_pi),
            torch.nn.ReLU(),
        )

        # Define the Critic network
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(features_dim, last_layer_dim_vf),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        """
        Forward pass for both Actor and Critic networks, used during testing.
        """
        actor_output = self.actor(x)
        critic_output = self.critic(x)
        return actor_output, critic_output

    def forward_actor(self, x):
        """
        Forward pass for the Actor network, used for policy prediction.
        """
        return self.actor(x)

    def forward_critic(self, x):
        """
        Forward pass for the Critic network, used for value function computation.
        """
        return self.critic(x)


class MyACPolicy(ActorCriticPolicy):
    def __init__(
        self,
        obs_space,
        action_space,
        lr_schedule,
        *args,
        **kwargs
    ):
        super().__init__(obs_space, action_space, lr_schedule, *args, **kwargs)
        self.ortho_init = False  # Disable orthogonal initialization (can be enabled based on requirements)

    def _build_mlp_extractor(self) -> None:
        # Initialize the custom MLP extractor
        self.mlp_extractor = MyACModel(self.features_dim)
        # Set the output dimensions for the Actor and Critic
        self.latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.latent_dim_vf = self.mlp_extractor.latent_dim_vf

    def _train(self):
        pass

# from typing import Callable
# import gymnasium as gym
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
#
#
# def train(
#         create_env: Callable[[], gym.Env],
#         vec_envs: int,
#         total_timesteps: int,
#         verbose: int = 0,
# ):
#     train_env = make_vec_env(create_env, n_envs=vec_envs)
#     train_env = create_env()
#     # model = A2C('MultiInputPolicy', env=train_env, verbose=verbose, n_steps=64, ent_coef=0.01)
#     model = PPO('MultiInputPolicy', train_env, verbose=verbose)
#     model.learn(total_timesteps=total_timesteps, progress_bar=True)
#     return model
#
# def test_env(create_env: Callable[[], gym.Env]):
#     vh_env = create_env()
#     vh_env.reset()
#     for _ in range(100):
#         obs = vh_env.action_space.sample()
#         print(obs)
#         print('--------------')
#
#
# def train_ppo_model():
#     from src.vh_env.food_gather import VirtualHomeGatherFoodEnvV2
#     from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
#
#     YOUR_FILE_NAME = "D:\\programs\\windows_exec.v2.2.4\\VirtualHome.exe"
#     comm = UnityCommunication(file_name=YOUR_FILE_NAME)
#     comm.reset(0)
#     comm.add_character()
#     res, g = comm.environment_graph()
#     comm.close()
#     print(g)
#
#     model = train(
#         create_env=lambda: VirtualHomeGatherFoodEnvV2(environment_graph=g, log_level='debug'),
#         vec_envs=4,
#         total_timesteps=500_000,
#         verbose=1
#     )
#     model.save("../../model/v2_first_test.zip")
#
# def test_ppo_model():
#     from src.vh_env.food_gather import VirtualHomeGatherFoodEnvV2
#     from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
#     YOUR_FILE_NAME = "D:\\programs\\windows_exec.v2.2.4\\VirtualHome.exe"
#     comm = UnityCommunication(file_name=YOUR_FILE_NAME)
#     comm.reset(0)
#     comm.add_character()
#     res, g = comm.environment_graph()
#     comm.close()
#     # print(g)
#
#     model = PPO.load("../../model/v2_first_test.zip")
#     vh_env = VirtualHomeGatherFoodEnvV2(
#         environment_graph=g,
#         log_level='warning'
#     )
#
#     obs, metadata = vh_env.reset()
#     done = False
#     while not done:
#         action, _ = model.predict(obs, deterministic=True)
#         action = int(action)
#         print(f'type{type(action)}, action:{vh_env.ACTION_LIST[action]}')
#         obs, reward, done, _, metadata = vh_env.step(action)
#     print(vh_env.get_instruction_list())
#
# def main2():
#     from src.vh_env.food_gather import VirtualHomeGatherFoodEnv
#     from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
#
#     YOUR_FILE_NAME = "D:\\programs\\windows_exec.v2.2.4\\VirtualHome.exe"
#     comm = UnityCommunication(file_name=YOUR_FILE_NAME)
#     comm.reset(0)
#     comm.add_character()
#     res, g = comm.environment_graph()
#     comm.close()
#
#     test_env(lambda: VirtualHomeGatherFoodEnv(environment_graph=g, log_level='debug'))
#
# if __name__ == "__main__":
#     # train_ppo_model()
#     # test_ppo_model()
#     pass