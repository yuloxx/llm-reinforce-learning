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
