"""第四篇的手算、独立求导与实际 TinyLM 缓存对照；不保存权重副本。"""
from collections import Counter
import argparse
import json
import math
import numpy as np
import torch
from .language import TinyLM
from .language_details import normalization_trace, sampling_trace, position_trace
from .mathlab import softmax


def byte_merges():
    # 频次跨三个样本累加；合并只在样本内部从左向右、不重叠地执行。
    samples = [list(b'abab'), list(b'abab'), list(b'abac')]
    vocabulary = {i: bytes([i]) for i in range(256)}
    history = []
    for new_id in [256, 257]:
        counts = Counter(pair for sample in samples for pair in zip(sample, sample[1:]))
        pair = min(counts, key=lambda p: (-counts[p], p))
        vocabulary[new_id] = vocabulary[pair[0]] + vocabulary[pair[1]]
        merged = []
        for sample in samples:
            output, index = [], 0
            while index < len(sample):
                if tuple(sample[index:index+2]) == pair:
                    output.append(new_id)
                    index += 2
                else:
                    output.append(sample[index])
                    index += 1
            merged.append(output)
        history.append({'pair': list(pair), 'count': counts[pair],
                        'new_id': new_id, 'piece': vocabulary[new_id].decode(),
                        'samples': merged})
        samples = merged
    assert [b''.join(vocabulary[i] for i in row) for row in samples] == [b'abab', b'abab', b'abac']
    return {'utf8_ids': list('苹果 AI'.encode()), 'steps': history}


def attention_backward():
    # 手工链式法则使用 NumPy；PyTorch 在独立计算图上给出对照梯度。
    x = np.array([[1., 0.], [0., 1.], [1., 1.]])
    scores = x @ x.T / math.sqrt(2)
    weights = softmax(scores)
    output = weights @ x
    upstream = np.zeros_like(output)
    upstream[0, 0] = output[0, 0] - 1
    dv = weights.T @ upstream
    da = upstream @ x.T
    ds = weights * (da - (weights * da).sum(axis=1, keepdims=True))
    dq, dk = ds @ x / math.sqrt(2), ds.T @ x / math.sqrt(2)
    dx = dq + dk + dv
    value = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    score = value @ value.T / math.sqrt(2)
    result = score.softmax(-1) @ value
    loss = (result[0, 0] - 1).square() / 2
    loss.backward()
    np.testing.assert_allclose(dx, value.grad.numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ds.sum(axis=1), 0., atol=1e-14)
    # 中心差分重新计算损失，不使用反向传播得到的中间量。
    finite = np.empty_like(x)
    epsilon = 1e-6
    def objective(v):
        prediction = softmax(v @ v.T / math.sqrt(2)) @ v
        return (prediction[0, 0] - 1)**2 / 2
    for index in np.ndindex(x.shape):
        positive, negative = x.copy(), x.copy()
        positive[index] += epsilon
        negative[index] -= epsilon
        finite[index] = (objective(positive) - objective(negative)) / (2*epsilon)
    np.testing.assert_allclose(dx, finite, rtol=1e-6, atol=1e-9)
    return {'x': x.tolist(), 'scores': scores.tolist(), 'weights': weights.tolist(),
            'output': output.tolist(), 'loss': float(loss.detach()),
            'd_output': upstream.tolist(), 'd_weights': da.tolist(), 'd_scores': ds.tolist(),
            'd_q': dq.tolist(), 'd_k': dk.tolist(), 'd_v': dv.tolist(), 'd_x': dx.tolist(),
            'finite_difference_max_error': float(np.abs(finite-dx).max())}


def heads_and_positions():
    # 位置仍按原序，头轴只拆分每个位置的特征列。
    x = torch.arange(12).reshape(1, 3, 4)
    heads = x.view(1, 3, 2, 2).transpose(1, 2)
    merged = heads.transpose(1, 2).contiguous().view(1, 3, 4)
    assert torch.equal(x, merged)
    def rotation(angle):
        return np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    q, k = np.array([1., 0.]), np.array([1., 1.])
    rotary = []
    for i, j in [(0, 1), (1, 2)]:
        left, right = rotation(i*math.pi/2)@q, rotation(j*math.pi/2)@k
        rotary.append({'positions': [i, j], 'q': left.tolist(), 'k': right.tolist(),
                       'dot': float(left@right)})
    np.testing.assert_allclose([item['dot'] for item in rotary], [-1., -1.], atol=1e-14)
    # 恒等残差与 F(x)=-x 相加，导数会抵消，直接项并不构成下界。
    scalar = torch.tensor(2., requires_grad=True)
    (scalar - scalar).backward()
    return {'input': x.tolist(), 'heads': heads.tolist(), 'head_stride': list(heads.stride()),
            'head_contiguous': heads.is_contiguous(), 'merged': merged.tolist(),
            'rotary': rotary, 'residual_cancellation_gradient': float(scalar.grad),
            'permutation': position_trace(), 'normalization': normalization_trace()}


def loss_and_step():
    # 概率梯度先手算，再让自动微分独立验证。
    probabilities = torch.tensor([.6, .2, .2], dtype=torch.float64)
    logits = probabilities.log().requires_grad_()
    loss = torch.nn.functional.cross_entropy(logits[None], torch.tensor([1]))
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([.6, -.8, .2], dtype=torch.float64))
    torch.manual_seed(17)
    model = TinyLM()
    ids = torch.tensor([list(b'read notes.\n')])
    inputs, targets = ids[:, :-1], ids[:, 1:]
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    optimizer.zero_grad(set_to_none=True)
    scores = model(inputs)
    initial = torch.nn.functional.cross_entropy(scores.reshape(-1, 256), targets.reshape(-1))
    initial.backward()
    gradients = {name: float(p.grad.norm()) for name, p in model.named_parameters()}
    first_row = model.embedding.weight[114].detach().clone()
    optimizer.step()
    with torch.no_grad():
        updated = torch.nn.functional.cross_entropy(model(inputs).reshape(-1, 256),
                                                    targets.reshape(-1))
    assert updated < initial
    assert sum(p.numel() for p in model.parameters()) == 25824
    return {'cross_entropy': float(loss.detach()), 'logit_gradient': logits.grad.tolist(),
            'inputs': inputs.tolist(), 'targets': targets.tolist(),
            'logits_shape': list(scores.shape), 'parameters': 25824,
            'initial_loss': float(initial.detach()), 'updated_loss': float(updated),
            'gradient_norms': gradients,
            'embedding_row_update_norm': float((model.embedding.weight[114]-first_row).detach().norm())}


def model_cache():
    # 对照完整 TinyLM，而不只比较注意力层；同时覆盖逐项和成块追加。
    torch.manual_seed(17)
    model = TinyLM().double().eval()
    ids = torch.tensor([list(b'read n')])
    with torch.no_grad():
        full = model(ids)
        past, parts, lengths = None, [], []
        for index in range(ids.shape[1]):
            scores, past, _ = model.forward_cached(ids[:, index:index+1], past)
            parts.append(scores)
            lengths.append(past[0].shape[-2])
        incremental = torch.cat(parts, dim=1)
        _, prefix, _ = model.forward_cached(ids[:, :2])
        block, _, block_weights = model.forward_cached(ids[:, 2:4], prefix)
        modified = ids.clone()
        modified[:, -1] = 120
        future_change = (model(ids)[:, :-1] - model(modified)[:, :-1]).abs().max().item()
        _, filled, _ = model.forward_cached(torch.zeros(1, 24, dtype=torch.long))
        try:
            model.forward_cached(torch.zeros(1, 1, dtype=torch.long), filled)
        except ValueError as error:
            rejection = str(error)
        else:
            raise AssertionError('位置表超限没有被拒绝')
    torch.testing.assert_close(full, incremental, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(full[:, 2:4], block, rtol=1e-12, atol=1e-12)
    assert future_change == 0
    return {'input_ids': ids.tolist(), 'cache_lengths': lengths,
            'full_incremental_max_error': float((full-incremental).abs().max()),
            'block_max_error': float((full[:, 2:4]-block).abs().max()),
            'block_visibility': [[True, True, True, False], [True, True, True, True]],
            'block_weights': block_weights.tolist(), 'future_change_error': future_change,
            'overflow_rejected': rejection, 'kv_float32_bytes_at_24': 6144}


def demo():
    torch.set_num_threads(2)
    return {'byte_merges': byte_merges(), 'attention_backward': attention_backward(),
            'heads_positions': heads_and_positions(), 'loss_step': loss_and_step(),
            'cache': model_cache(), 'sampling': sampling_trace()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='实际训练并比较共同保留集')
    args = parser.parse_args()
    if args.train:
        from .language import train_language, combination_experiment
        result = {'baseline': train_language(), 'combination': combination_experiment()}
    else:
        result = demo()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
