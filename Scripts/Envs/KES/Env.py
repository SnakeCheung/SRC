import torch
import torch.nn as nn
from .utils import load_d_agent, episode_reward

class KESEnv():
    def __init__(self, dataset, model_name='DKT', dataset_name='assist09', device = 'cuda'):
        if dataset_name == 'assist09':
            self.skill_num = 123
        elif dataset_name == 'assist12':
            self.skill_num = 265
        elif dataset_name == 'assist15':
            self.skill_num = 100
        elif dataset_name == 'assist17':
            self.skill_num = 102
        elif dataset_name == 'algebra2005':
            self.skill_num = 112
        elif dataset_name == 'bridge2006':
            self.skill_num = 493
        elif dataset_name == 'ednet':
            self.skill_num = 188
        elif dataset_name == 'junyi':
            self.skill_num = 39
        elif dataset_name == 'nips34':
            self.skill_num = 57
        elif dataset_name == 'xes3g5m':
            self.skill_num = 865
        elif dataset_name == 'mooccubex':
            self.skill_num = 438
        self.device = device
        self.model = load_d_agent(model_name, dataset_name, self.skill_num)
        self.model.to(self.device)
        self.targets = None
        self.states = (None, None)
        self.initial_score = None

    def exam(self, targets, states):
        """
        评估学生对目标知识点的掌握程度
        """
        scores = []
        # targets 形状通常为 (B, K)
        for i in range(targets.size(1)):
            # 假设 model.learn_lstm 返回 (score, next_states)
            # targets[:, i:i+1] 保持维度为 (B, 1)
            score, _ = self.model.learn_lstm(targets[:, i:i + 1], *states)
            scores.append(score)
        
        # scores 列表拼接后形状为 (B, K)，在 dim=1 上求均值得到 (B,)
        return torch.mean(torch.cat(scores, dim=1), dim=1)

    def begin_episode(self, targets, initial_logs):
        """
        开始一个新的教学回合
        """
        self.targets = targets
        initial_score, initial_log_scores, states = self.begin_episode_(targets, initial_logs)
        self.initial_score = initial_score
        self.states = states
        return initial_log_scores

    def begin_episode_(self, targets, initial_logs=None):
        """
        内部逻辑：初始化状态并计算初始得分
        """
        states = (None, None)
        score = None
        if initial_logs is not None:
            # 获取学生初始的知识状态
            score, states = self.model.learn_lstm(initial_logs)
            
        # 计算学生在当前知识状态下，对目标知识点的初始掌握率
        initial_score = self.exam(targets, states)
        return initial_score, score, states

    def n_step(self, learning_path, binary=False):
        """
        执行教学步（学习路径），更新学生状态
        """
        # learning_path: (B, L)
        scores, states = self.model.learn_lstm(learning_path, *self.states)
        self.states = states
        
        if binary:
            # PyTorch 中将布尔值转为 float32 的标准做法
            scores = (scores > 0.5).float()
            
        return scores

    def end_episode(self, **kwargs):
        """
        结束回合并计算奖励
        """
        final_score, reward = self.end_episode_(self.initial_score, self.targets, *self.states)
        if 'score' in kwargs:
            return final_score, reward
        return reward

    def end_episode_(self, initial_score, targets, states1, states2):
        """
        内部逻辑：计算最终得分和奖励值
        """
        final_score = self.exam(targets, (states1, states2))
        
        # 假设 episode_reward 已处理为 PyTorch 兼容
        reward = episode_reward(initial_score, final_score, 1).unsqueeze(-1)
        return final_score, reward