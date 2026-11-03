"""第四篇的逐项计算、条件对照和缓存错误定位。"""
import json
import math
import sys
import numpy as np
import torch
from .mathlab import attention, softmax
from .networks import CausalAttention, cache_experiment


def position_trace():
    # 未加掩码时，交换输入行会同步交换输出；位置向量留在原来的位置。
    x = np.array([[1., 0.], [0., 1.], [1., 1.]])
    order = [1, 0, 2]
    positions = np.array([[0., 0.], [0., 1.], [1., 0.]])
    plain = attention(x, x, x)[2]
    moved = attention(x[order], x[order], x[order])[2]
    located = x+positions
    relocated = x[order]+positions
    with_position = attention(located, located, located)[2]
    moved_position = attention(relocated, relocated, relocated)[2]
    def rotation(angle):
        return np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    # 两对位置相差都为 1，角频率 pi/2；同移后点积仍不变。
    vector = np.array([1., 0.])
    pairs = []
    for i, j in [(0, 1), (1, 2)]:
        q, k = rotation(i*math.pi/2)@vector, rotation(j*math.pi/2)@vector
        pairs.append({'positions': [i, j], 'q': q.tolist(), 'k': k.tolist(),
                      'dot': float(q@k)})
    return {'unmasked_permutation_error': float(np.abs(moved-plain[order]).max()),
            'fixed_position_permutation_error': float(np.abs(moved_position-with_position[order]).max()),
            'rotary_pairs': pairs}


def cache_failure():
    # 正确缓存来自正式注意力层；错误分支仅故意漏掉可见范围中的历史偏移。
    torch.manual_seed(17)
    layer = CausalAttention().double().eval()
    values = torch.randn(1, 5, 16, dtype=torch.float64)
    cache, differences, lengths = None, [], []
    with torch.no_grad():
        for index in range(values.shape[1]):
            current = values[:, index:index+1]
            correct, cache, _ = layer(current, cache)
            q = layer.qkv(current).chunk(3, dim=-1)[0]
            q = q.reshape(1, 1, layer.heads, -1).transpose(1, 2)
            scores = q@cache[0].transpose(-2, -1)/math.sqrt(layer.width//layer.heads)
            # 当前块只有一个查询，局部下标为 0；漏偏移就只能读取键 0。
            wrong_visible = torch.arange(cache[0].shape[-2])[None, :] <= 0
            wrong_weights = scores.masked_fill(~wrong_visible, -torch.inf).softmax(-1)
            mixed = (wrong_weights@cache[1]).transpose(1, 2).reshape(1, 1, layer.width)
            wrong = layer.out(mixed)
            differences.append(float((correct-wrong).abs().max()))
            lengths.append(cache[0].shape[-2])
    return {'correct': cache_experiment(), 'cache_lengths': lengths,
            'wrong_offset_errors': differences,
            'first_wrong_position': next(i for i, error in enumerate(differences) if error>1e-12),
            'position_ablation': position_trace()}


def normalization_trace():
    # 方差除以特征数 4，与 nn.LayerNorm 的定义一致。
    x = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
    norm = torch.nn.LayerNorm(4, eps=1e-5).double()
    return {'mean': float(x.mean()), 'variance': float(x.var(unbiased=False)),
            'normalized': norm(x).detach().tolist(),
            'gelu_positive_negative': torch.nn.functional.gelu(
                torch.tensor([1., -1.], dtype=torch.float64)).tolist()}


def sampling_trace():
    # 三个候选的原始指数比例为 4:2:1。
    logits = np.log([4., 2., 1.])
    full = softmax(logits)
    return {'logits': logits.tolist(), 'temperature_1': full.tolist(),
            'temperature_half': softmax(logits/.5).tolist(),
            'top_k_2': [2/3, 1/3, 0.],
            'top_p_07_kept': [0, 1], 'top_p_07_probabilities': [2/3, 1/3, 0.]}


def adaptation_trace():
    # 主路径与低秩路径使用正文同一组矩阵，完整保留两个中间向量。
    x = np.array([1., 2., 3., 4.])
    w = np.eye(3, 4)
    a = np.array([[1., 0., 1., 0.], [0., 1., 0., 1.]])
    b = np.array([[1., 0.], [0., 1.], [1., -1.]])
    base, narrow, update = w@x, a@x, b@a@x
    # 对称 int8 取 -127 到 127；反量化后的误差要回到原浮点值上比较。
    weights = np.array([-1., -.2, .3, 1.])
    scale = float(np.abs(weights).max()/127)
    quantized = np.clip(np.rint(weights/scale), -127, 127).astype(np.int8)
    restored = scale*quantized.astype(float)
    # DPO 比较相对参考策略的完整回答概率比。
    beta = .5
    margin = beta*(math.log(.4/.2)-math.log(.1/.2))
    return {'lora': {'base': base.tolist(), 'narrow': narrow.tolist(),
                     'update': update.tolist(), 'output': (base+update).tolist()},
            'quantization': {'weights': weights.tolist(), 'scale': scale,
                             'integers': quantized.tolist(), 'restored': restored.tolist(),
                             'max_error': float(np.abs(restored-weights).max())},
            'dpo': {'beta': beta, 'margin': margin, 'preference_probability': 1/(1+math.exp(-margin)),
                    'loss': math.log1p(math.exp(-margin))},
            'ppo': {'old_probability': .2, 'new_probability': .3,
                    'ratio': .3/.2, 'clipped_ratio': 1.2,
                    'positive_advantage_term': min(.3/.2, 1.2),
                    'negative_advantage_term': min(-.3/.2, -1.2)}}


def supervision_trace():
    # 在 Q:a / A:b 中，只有回答 b 的预测位置进入 SFT 损失。
    sequence = list(b'Q:a\nA:b')
    inputs, targets = sequence[:-1], sequence[1:]
    labels = [-100]*(len(targets)-1)+[targets[-1]]
    return {'sequence': sequence, 'inputs': inputs, 'targets': targets,
            'answer_only_labels': labels, 'answer_target_count': 1,
            'uniform_byte_loss': math.log(256)}


def demo():
    # 全部小计算只保留内存状态，输出可与正文、配图逐项对照。
    return {'position': position_trace(), 'normalization': normalization_trace(),
            'sampling': sampling_trace(), 'adaptation': adaptation_trace(),
            'supervision': supervision_trace(), 'cache_failure': cache_failure()}


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps(demo(), ensure_ascii=False, indent=2))
