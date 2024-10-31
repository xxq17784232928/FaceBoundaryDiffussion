import torch

def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    计算两个向量 x 和 y 之间的欧几里得距离。

    参数:
    x (torch.Tensor): 输入向量，形状为 (n,) 或 (batch_size, n)。
    y (torch.Tensor): 输入向量，形状为 (n,) 或 (batch_size, n)。

    返回:
    torch.Tensor: 欧几里得距离，形状为 (1,) 或 (batch_size, 1)。
    """
    # 计算向量差
    diff = x - y
    
    # 计算平方差并求和，保留维度以避免降维
    dist_squared = torch.sum(diff ** 2, dim=-1, keepdim=True)
    
    # 返回距离，取平方根
    dist = torch.sqrt(dist_squared)
    
    return dist


def cosine_similarity(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    计算两个向量 x 和 y 之间的余弦相似度。
    
    参数:
    x (torch.Tensor): 输入向量，形状为 (n,) 或 (batch_size, n)。
    y (torch.Tensor): 输入向量，形状为 (n,) 或 (batch_size, n)。
    eps (float): 防止除以零的小数（默认: 1e-8）。
    
    返回:
    torch.Tensor: 余弦相似度，形状为 (1,) 或 (batch_size, 1)。
    """
    # 计算向量点积
    dot_product = torch.sum(x * y, dim=-1, keepdim=True)
    
    # 计算向量的模长 (L2 范数)
    norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
    norm_y = torch.norm(y, p=2, dim=-1, keepdim=True)
    
    # 计算余弦相似度，加入 eps 以防除以 0 的情况
    cosine_sim = dot_product / (norm_x * norm_y).clamp(min=eps)
    
    return cosine_sim