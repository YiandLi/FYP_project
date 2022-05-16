import torch
import numpy as np

# print(sys.path)
# sys.path.insert(0, '/Users/liuyilin/Downloads/SemiEval 2021/Session5')


# 直接用entropy来衡量 自信度： entropy 越低越自信；
def normal_shannon_entropy(p, labels_num):
    p = torch.tensor(p)
    p = torch.softmax(p, dim=-1)
    
    entropy = torch.distributions.Categorical(probs=p).entropy()
    # == log(labels_num)
    normal = -np.log(1.0 / labels_num)
    # normal = 1
    return (entropy / normal).numpy()


def can_adjustment(predicted_probs,
                   priors=None,
                   alpha=1,
                   iters=3,
                   threshold=0.99,
                   ):
    """
    :param predicted_probs: 拼接的矩阵 batch_size/instance_num * m
    :param priors:  先验 m维向量
    :param alpha:   放缩参数 L^alpha
    :param iters:   迭代次数
    :param threshold:   选择的阈值
    :return:    batch_size/instance_num * m
    """
    preds_entropy = normal_shannon_entropy(predicted_probs, predicted_probs.shape[-1])
    
    # 根据阈值，找出高置信度样本
    preds_confident = predicted_probs[preds_entropy < threshold]
    
    # priors
    priors = np.array(priors, dtype=np.float)
    priors /= priors.sum(keepdims=True, axis=-1)
    
    # 现在遍历所有样本, 自信的样本保留原样，不自信样本进行调整；
    preds_adjusted = []
    for i in range(predicted_probs.shape[0]):
        if preds_entropy[i] <= threshold:
            preds_adjusted.append(predicted_probs[i])
        
        else:
            p_combined = np.append(preds_confident, predicted_probs[i][None], axis=0)
            
            for j in range(iters):
                # print("*** row normalizing ***")
                p_combined = p_combined ** alpha
                p_combined /= p_combined.sum(axis=0, keepdims=True)
                
                # print("*** column normalizing ***")
                p_combined *= priors.reshape(1, -1)
                p_combined /= p_combined.sum(axis=1, keepdims=True)
            
            pred_ = p_combined[-1]
            preds_adjusted.append(pred_)
    
    preds_adjusted = np.stack(preds_adjusted, axis=0)
    
    return preds_adjusted


if __name__ == "__main__":
    # 类别数量
    K = 3
    ## TODO: 构造样本矩阵
    # 随机样本 【6，3】
    a = np.random.uniform(0, 10, (6, K))
    # 归一为概率形式
    a /= a.sum(keepdims=True, axis=-1)
    # 人为构造一个低置信度样本 【1，3】
    b = np.array([1 / K for _ in range(K)]).reshape(1, -1)
    predicted_probs = np.append(a, b, axis=0)
    print("predicted_probs: ", predicted_probs.shape)
    print(predicted_probs)
    
    #######################
    # implement CAN method: CAN概率校准方法
    #######################
    # 先验分布
    priors = np.array([1 / K for _ in range(K)], dtype=np.float)
    
    preds_adjusted = can_adjustment(predicted_probs,
                                    priors=priors,
                                    alpha=3,
                                    iters=1,
                                    threshold=0.995,
                                    )
    
    print("\nafter adjusted: ")
    print(preds_adjusted)