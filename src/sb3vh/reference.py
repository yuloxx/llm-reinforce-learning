from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    create_mlp,
)
import torch as th
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer  # 示例导入，根据实际LLM调整


# 自定义网络结构，这里继承BaseFeaturesExtractor方便处理输入特征
class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        # 这里自定义具体的网络层，例如简单的多层感知机示例
        self.layers = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, features_dim),
        )

    def forward(self, observations):
        return self.layers(observations)


# 自定义策略类，继承ActorCriticPolicy
class CustomPolicyWithLLMEvaluation(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super(CustomPolicyWithLLMEvaluation, self).__init__(
            observation_space, action_space, lr_schedule, *args, **kwargs
        )
        # 使用自定义特征提取器
        self.features_extractor = CustomFeatureExtractor(observation_space)
        self.features_dim = self.features_extractor.features_dim

        # 策略网络（actor）和价值网络（critic）示例结构，可根据需求调整
        self.mlp_extractor = create_mlp(self.features_dim, self.action_space.n, net_arch=[32])
        self.value_net = create_mlp(self.features_dim, 1, net_arch=[32])

        # 加载LLM相关模型和tokenizer（这里以一个示例的因果语言模型为例，需替换为实际的）
        self.llm_model = AutoModelForCausalLM.from_pretrained("gpt2")  # 根据实际LLM调整
        self.llm_tokenizer = AutoTokenizer.from_pretrained("gpt2")  # 根据实际LLM调整

    def forward(self, obs, deterministic=True):
        """
        前向传播，计算动作和价值估计
        """
        features = self.extract_features(obs)
        action_logits = self.mlp_extractor(features)
        values = self.value_net(features).flatten()
        return action_logits, values

    def evaluate_actions(self, obs, actions):
        """
        评估动作，同时引入LLM评估
        """
        features = self.extract_features(obs)
        action_logits, values = self.forward(obs)
        # 计算动作对数概率等常规操作（示例简化）
        action_log_probs = action_logits.log_softmax(dim=-1)
        dist_entropy = action_log_probs.mean()

        # 准备输入给LLM进行评估，这里示例将观测和动作转换为文本描述（需根据实际合理设计）
        obs_text = self.observation_to_text(obs)
        action_text = self.action_to_text(actions)
        input_text = f"Observation: {obs_text}\nAction: {action_text}\nEvaluate the action:"
        input_ids = self.llm_tokenizer.encode(input_text, return_tensors="pt")

        # 获取LLM输出并解析评估结果（示例简化，实际需根据LLM输出合理解析）
        with th.no_grad():
            output = self.llm_model.generate(input_ids)
            llm_evaluation_text = self.llm_tokenizer.decode(output[0], skip_special_tokens=True)
            llm_evaluation_score = self.parse_llm_evaluation(llm_evaluation_text)

        # 根据LLM评估结果调整相关指标（示例简化，可能需要更合理设计权重等）
        adjusted_action_log_probs = action_log_probs * (1 + llm_evaluation_score)
        return adjusted_action_log_probs, values, dist_entropy

    def observation_to_text(self, obs):
        """
        将观测转换为适合LLM输入的文本描述，需根据实际情况定制
        """
        # 示例，简单将数值观测转为文本描述
        return " ".join([f"feature_{i}: {val}" for i, val in enumerate(obs.flatten())])

    def action_to_text(self, actions):
        """
        将动作转换为适合LLM输入的文本描述，需根据实际情况定制
        """
        # 示例，简单描述动作
        return f"Action value: {actions}"

    def parse_llm_evaluation(self, text):
        """
        解析LLM输出的文本评估结果，返回一个数值分数用于后续调整（示例简化）
        """
        # 假设LLM输出类似 "The action is good with a score of 0.8" 这样的文本
        # 提取其中的分数，实际需更健壮的解析
        score_str = text.split("score of ")[-1].split(" ")[0]
        return float(score_str)


