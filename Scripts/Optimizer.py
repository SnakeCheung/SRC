import torch
import torch.nn as nn

class ModelWithLoss(nn.Module):
    def __init__(self, model, criterion):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, *data):
        """
        对应 MindSpore 的 construct。
        通常在采样（Sampling）阶段使用。
        """
        # 最后一个参数是 rewards，前面的是输入数据
        inputs, rewards = data[:-1], data[-1]
        output_data = self.model(*inputs)
        
        # 根据之前的 SRC 模型，output_data[1] 是 probs (概率)
        return self.criterion(output_data[1], rewards)

    def backup(self, *data):
        """
        对应 MindSpore 的 backup。
        通常在更新（Update）阶段使用，计算 Teacher Forcing 下的概率。
        """
        inputs, rewards = data[:-1], data[-1]
        # 调用 model 的 backup 方法获取概率
        output_data = self.model.backup(*inputs)
        return self.criterion(output_data, rewards)


class ModelWithOptimizer(nn.Module):
    def __init__(self, model_with_loss, optimizer):
        super(ModelWithOptimizer, self).__init__()
        self.model_with_loss = model_with_loss
        self.optimizer = optimizer

    def forward(self, *data):
        """
        执行完整的训练步：前向传播 -> 反向传播 -> 梯度裁剪 -> 参数更新
        """
        # 1. 清空梯度
        self.optimizer.zero_grad()
        
        # 2. 调用 model_with_loss 的 backup 方法计算 loss
        # 注意：这里对应原代码中 grad_fn 使用的是 backup 方法
        loss = self.model_with_loss.backup(*data)
        
        # 3. 反向传播计算梯度
        loss.backward()
        
        # 4. 全局梯度裁剪 (Global Norm Clipping)
        # MindSpore 的 clip_by_global_norm(grads, 20)
        torch.nn.utils.clip_grad_norm_(self.model_with_loss.parameters(), max_norm=20)
        
        # 5. 更新参数
        self.optimizer.step()
        
        return loss