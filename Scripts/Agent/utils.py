import torch
import numpy as np

def pl_loss(pro, reward):
    """
    Policy Gradient Loss (Negative Log Likelihood weighted by reward)
    pro: probability of the selected action
    reward: the reward/advantage for that action
    """
    # Equivalent to MindSpore: -ops.mean(reward * ops.log(pro + 1e-9))
    return -torch.mean(reward * torch.log(pro + 1e-9))


def generate_path(batch_size, skill_num, path_type, n):
    """
    Generates synthetic paths/sequences of skills.
    """
    if path_type in (0, 1):
        # Generate a random permutation of indices from 0 to n-1 for each batch
        origin_path = np.argsort(np.random.rand(batch_size, n))
        
        if path_type == 1:
            # Grouping: offset the indices to a specific 'block' of skills
            offset = n * np.random.randint(0, skill_num // n, (batch_size, 1))
            origin_path += offset
            
    else:  # path_type 2 or 3
        # Randomly sort all available skills
        origin_path = np.argsort(np.random.rand(batch_size, skill_num))
        
        if path_type == 2:
            # Take only the top N from the random sort
            origin_path = origin_path[:, :n]
            
    return torch.from_numpy(origin_path.astype(np.int32))