"""手算梯度、循环状态、卷积形状与因果注意力的 PyTorch 对照。"""
import math
import torch
from torch import nn


def verify_gradients():
    # 叶子参数 w 的梯度应等于正文手算的 60。
    w = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    prediction = 2 * torch.relu(w * 3 + 1)
    loss = (prediction - 4).square() / 2
    loss.backward()
    return {'loss': loss.item(), 'dw': w.grad.item()}


class CausalAttention(nn.Module):
    def __init__(self, width=16, heads=2):
        super().__init__()
        if width % heads:
            raise ValueError('width 必须能够均分到 heads 个头')
        self.width, self.heads = width, heads
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, x, past=None):
        # 每个头保存自己的 K、V；旧查询不必重新计算。
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(t):
            return t.view(batch, length, self.heads, -1).transpose(1, 2)
        q, k, v = map(split, (q, k, v))
        old = 0 if past is None else past[0].shape[-2]
        if past is not None:
            k, v = torch.cat((past[0], k), dim=-2), torch.cat((past[1], v), dim=-2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.width // self.heads)
        visible = torch.arange(k.shape[-2], device=x.device)[None, :] <= old + torch.arange(length, device=x.device)[:, None]
        weights = scores.masked_fill(~visible, -torch.inf).softmax(dim=-1)
        mixed = (weights @ v).transpose(1, 2).contiguous().view(batch, length, self.width)
        return self.out(mixed), (k, v), weights


def cache_experiment():
    # 完整序列与逐 Token 缓存应给出相同的注意力输出。
    torch.manual_seed(17)
    layer = CausalAttention().double().eval()
    x = torch.randn(1, 5, 16, dtype=torch.float64)
    with torch.no_grad():
        full, _, weights = layer(x)
        cache, parts, lengths = None, [], []
        for index in range(x.shape[1]):
            value, cache, _ = layer(x[:, index:index + 1], cache)
            parts.append(value)
            lengths.append(cache[0].shape[-2])
        incremental = torch.cat(parts, dim=1)
    return {'max_error': (full - incremental).abs().max().item(), 'cache_lengths': lengths,
            'future_mass': weights[0, 0].triu(1).sum().item()}


def shapes():
    # 固定输入，核对 CNN、RNN、LSTM 和 GRU 的轴与输出位置。
    torch.manual_seed(17)
    image = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
    conv = nn.Conv2d(1, 2, 3, stride=1, padding=0)
    sequence = torch.randn(2, 4, 3)
    result = {'image': list(image.shape), 'conv': list(conv(image).shape)}
    for name, cell in [('rnn', nn.RNN(3, 5, batch_first=True)),
                       ('lstm', nn.LSTM(3, 5, batch_first=True)),
                       ('gru', nn.GRU(3, 5, batch_first=True))]:
        output, state = cell(sequence)
        result[name] = {'output': list(output.shape), 'hidden': list((state[0] if isinstance(state, tuple) else state).shape)}
    return result


def temporal_gradients():
    # 参数 r 在三个时间步共享，梯度累加各条路径。
    r = torch.tensor(.5, dtype=torch.float64, requires_grad=True)
    h, states = torch.tensor(0., dtype=torch.float64), []
    for value in [1., 2., 0.]:
        h = value + r*h
        states.append(h.item())
    loss = (h-1).square()/2
    loss.backward()
    # 从末端往前走，各步对共享参数 r 的贡献相加。
    old_states = [0.] + states[:-1]
    path_contributions = [(states[-1]-1)*.5**(2-index)*previous
                          for index, previous in enumerate(old_states)]
    # 同一递推展开为 h3=r²+2r；中心差分独立重算两次损失。
    polynomial_gradient = (states[-1]-1)*(2*.5+2)
    def scalar_loss(value):
        return (value**2+2*value-1)**2/2
    epsilon = 1e-6
    finite_difference = (scalar_loss(.5+epsilon)-scalar_loss(.5-epsilon))/(2*epsilon)
    # PyTorch LSTM 门顺序为 i、f、g、o，用偏置得到指定门值。
    cell = nn.LSTMCell(1, 1).double()
    with torch.no_grad():
        for parameter in cell.parameters():
            parameter.zero_()
        cell.bias_ih.copy_(torch.tensor([math.log(.2/.8), math.log(.8/.2),
                                        math.atanh(.5), math.log(.75/.25)], dtype=torch.float64))
        hidden, memory = cell(torch.zeros(1, 1, dtype=torch.float64),
                              (torch.zeros(1, 1, dtype=torch.float64),
                               torch.ones(1, 1, dtype=torch.float64)))
    return {'states': states, 'loss': loss.item(), 'dr': r.grad.item(),
            'path_contributions': path_contributions,
            'polynomial_gradient': polynomial_gradient,
            'epsilon': epsilon, 'finite_difference': finite_difference,
            'lstm_c': memory.item(), 'lstm_h': hidden.item(),
            'gru_h': .8 + .2*math.tanh(.5)}


def sequence_gradient_norms(lengths=(3, 12)):
    # 线性 RNN h_t=x_t+0.5h_{t-1}；末端读出对自身的梯度固定为 1。
    results = []
    for length in lengths:
        inputs = torch.ones(length, dtype=torch.float64, requires_grad=True)
        hidden = torch.zeros((), dtype=torch.float64)
        states = []
        for value in inputs:
            hidden = value + .5*hidden
            hidden.retain_grad()  # 保留非叶子状态收到的反向梯度。
            states.append(hidden)
        hidden.backward()
        # 标量状态的欧氏范数就是梯度绝对值。
        norms = [abs(state.grad.item()) for state in states]
        results.append({'length': length, 'state_gradient_norms': norms,
                        'input_gradient_norms': inputs.grad.abs().tolist(),
                        'first_over_last': norms[0]/norms[-1]})
    return results


def sequence_training():
    # 末端 Token 恒为 2；类别完全取决于最早的 0 或 1。
    torch.manual_seed(17)
    torch.set_num_threads(2)
    embedding, recurrent, head = nn.Embedding(4, 8), nn.RNN(8, 12, batch_first=True), nn.Linear(12, 2)
    parameters = list(embedding.parameters()) + list(recurrent.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(parameters, lr=.03)
    x = torch.tensor([[0, 3, 2], [1, 3, 2], [0, 0, 2], [1, 0, 2]])
    y = torch.tensor([0, 1, 0, 1])
    losses = []
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        output, _ = recurrent(embedding(x))
        loss = nn.functional.cross_entropy(head(output[:, -1]), y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        output, _ = recurrent(embedding(x))
        predictions = head(output[:, -1]).argmax(-1).tolist()
    return {'initial_loss': losses[0], 'final_loss': losses[-1], 'predictions': predictions,
            'last_hidden_first_pair': output[:2, -1].tolist(),
            'gradient_norm': float(torch.linalg.vector_norm(recurrent.weight_hh_l0.grad))}


def training_diagnostics():
    # 连续反向两次会累积；清空后恢复单次贡献。
    p = torch.tensor(1., requires_grad=True)
    (p*3).backward()
    first = p.grad.item()
    (p*3).backward()
    accumulated = p.grad.item()
    p.grad = None
    (p*3).backward()
    dropout = nn.Dropout(.5)
    torch.manual_seed(17)
    train = dropout(torch.ones(8)).tolist()
    dropout.eval()
    return {'first': first, 'accumulated': accumulated, 'cleared': p.grad.item(),
            'dropout_train': train, 'dropout_eval': dropout(torch.ones(8)).tolist()}


def demo():
    # 所有训练结果只留在内存，不生成权重与数据副本。
    return {'gradient': verify_gradients(), 'temporal': temporal_gradients(),
            'shapes': shapes(), 'cache': cache_experiment(),
            'sequence': sequence_training(), 'diagnostics': training_diagnostics()}
