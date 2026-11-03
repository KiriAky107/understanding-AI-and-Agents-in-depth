"""第一至三篇的手算对应实现；所有数组使用双精度核对。"""
import numpy as np


def softmax(scores, axis=-1):
    # 减去最大值，保持概率比值并避免指数溢出。
    scores = np.asarray(scores, dtype=float)
    values = np.exp(scores - np.max(scores, axis=axis, keepdims=True))
    return values / values.sum(axis=axis, keepdims=True)


def cosine(a, b):
    # 零向量没有方向，调用者应先处理没有特征的记录。
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = np.linalg.norm(a) * np.linalg.norm(b)
    if length == 0:
        return 0.0
    return float(a @ b / length)


def attention(q, k, v, causal=False):
    # 返回分数、概率和输出，便于逐项检查信息从哪里来。
    q, k, v = (np.asarray(x, float) for x in (q, k, v))
    scores = q @ k.T / np.sqrt(q.shape[-1])
    if causal:
        scores = np.where(np.tril(np.ones(scores.shape, dtype=bool)), scores, -np.inf)
    weights = softmax(scores)
    return scores, weights, weights @ v


def backprop_example(w=2.0):
    # x=3、目标=4，隐藏层 z=wx+1，输出=2*ReLU(z)。
    x, target = 3.0, 4.0
    z = w * x + 1
    h = max(z, 0)
    prediction = 2 * h
    loss = (prediction - target) ** 2 / 2
    grad = (prediction - target) * 2 * (z > 0) * x
    return {'z': z, 'h': h, 'prediction': prediction, 'loss': loss, 'dw': grad}


def gradient_check(epsilon=1e-6):
    # 中心差分用左右两个损失近似同一点的斜率。
    return (backprop_example(2 + epsilon)['loss'] - backprop_example(2 - epsilon)['loss']) / (2 * epsilon)


def rnn_walk(values=(1, 2, 0), recurrent=0.5):
    # 线性状态更新便于手算；非线性 RNN 在网络模块中实现。
    state, steps = 0.0, []
    for value in values:
        old = state
        state = value + recurrent * state
        steps.append({'input': value, 'before': old, 'after': state})
    return steps


def pca_example():
    # 两列完全相关；主方向保存沿对角线的全部变化。
    x = np.array([[1, 2], [2, 4], [3, 6]], dtype=float)
    mean = x.mean(axis=0)
    u, singular, vt = np.linalg.svd(x - mean, full_matrices=False)
    component = vt[0]
    if component[0] < 0:
        component = -component
    scores = (x - mean) @ component
    restored = scores[:, None] * component + mean
    return {'mean': mean.tolist(), 'singular': singular.tolist(), 'component': component.tolist(),
            'scores': scores.tolist(), 'restored': restored.tolist()}


def clip_loss(images, texts, temperature=1.0):
    # 两个方向都把对角线配对视为正确类别。
    images, texts = np.asarray(images, float), np.asarray(texts, float)
    images = images / np.linalg.norm(images, axis=1, keepdims=True)
    texts = texts / np.linalg.norm(texts, axis=1, keepdims=True)
    logits = images @ texts.T / temperature
    image_prob = softmax(logits, axis=1)
    text_prob = softmax(logits, axis=0)
    loss = -(np.log(np.diag(image_prob)).mean() + np.log(np.diag(text_prob)).mean()) / 2
    return {'logits': logits.tolist(), 'image_prob': image_prob.tolist(), 'text_prob': text_prob.tolist(), 'loss': float(loss)}


def demo():
    # 输出与正文逐项对应的数字，数组统一转换为 JSON 可写入类型。
    q = [[1, 0]]
    k, v = [[1, 0], [0, 1]], [[2, 0], [0, 4]]
    scores, weights, output = attention(q, k, v)
    return {'dot': float(np.array([3, 4]) @ np.array([4, 0])),
            'cosine': cosine([3, 4], [4, 0]), 'projection': [3, 0],
            'backprop': backprop_example(), 'finite_difference': gradient_check(),
            'rnn': rnn_walk(), 'pca': pca_example(),
            'attention': {'scores': scores.tolist(), 'weights': weights.tolist(), 'output': output.tolist()},
            'clip': clip_loss(np.eye(2), np.eye(2))}
