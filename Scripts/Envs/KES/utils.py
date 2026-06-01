import os
import torch

# 假设这些工具函数也已经适配了 PyTorch
from longling import path_append, abs_current_dir
from KTScripts.options import get_exp_configure
from KTScripts.utils import load_model


def load_d_agent(model_name, dataset_name, skill_num, with_label=True):
    # 1. 获取配置参数
    model_parameters = get_exp_configure(model_name)
    model_parameters.update({
        'feat_nums': skill_num, 
        'model': model_name, 
        'without_label': not with_label
    })
    
    if model_name == 'GRU4Rec':
        model_parameters.update({'output_size': skill_num})
    
    # 2. 加载模型结构 (假设 load_model 返回的是 nn.Module)
    model = load_model(model_parameters)
    
    # 3. 构建权重路径
    model_folder = path_append(abs_current_dir(__file__), os.path.join('meta_data'))
    model_path = os.path.join(model_folder, f'{model_name}_{dataset_name}')
    if not with_label:
        model_path += '_without'
    
    # 4. 加载权重
    # 注意：MindSpore 使用 .ckpt，PyTorch 通常使用 .pth 或 .pt
    # 这里保留路径逻辑，你可以根据实际文件名调整后缀
    ckpt_full_path = f'{model_path}.pth' 
    if not os.path.exists(ckpt_full_path):
        ckpt_full_path = f'{model_path}.ckpt' # 兼容原有的后缀名
        
    state_dict = torch.load(ckpt_full_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    # 5. 设置为评估模式 (等同于 MindSpore 的 set_train(False))
    model.eval()
    
    return model


def episode_reward(initial_score, final_score, full_score):
    """
    计算归一化的奖励值
    """
    delta = final_score - initial_score
    # 1e-9 为了防止除以 0
    normalize_factor = full_score - initial_score + 1e-9
    return delta / normalize_factor