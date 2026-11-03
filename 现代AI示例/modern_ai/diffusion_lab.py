"""4×4 图样上的 DDPM；显式保存系数和每次反向转移，不保存权重文件。"""
from copy import deepcopy
import hashlib
import math
import numpy as np
import torch
from torch import nn


def diffusion_trace():
    alpha = np.array([.9, .8, .7])
    cumulative = np.cumprod(alpha)
    clean, epsilon, predicted = .8, -.6, -.5
    noisy = .8*clean+.6*epsilon
    estimate = (noisy-.6*predicted)/.8
    alpha_t, bar_t, bar_previous = .8, .72, .9
    beta_t = 1-alpha_t
    posterior_variance = beta_t*(1-bar_previous)/(1-bar_t)
    # 对同一给定 x_t、x0，完成平方的均值与噪声参数化均值一致。
    x0, xt = .8, .28
    exact_epsilon = (xt-math.sqrt(bar_t)*x0)/math.sqrt(1-bar_t)
    mean_from_x0 = (math.sqrt(bar_previous)*beta_t/(1-bar_t)*x0+
                    math.sqrt(alpha_t)*(1-bar_previous)/(1-bar_t)*xt)
    mean_from_noise = (xt-beta_t/math.sqrt(1-bar_t)*exact_epsilon)/math.sqrt(alpha_t)
    np.testing.assert_allclose(mean_from_x0, mean_from_noise, atol=1e-14)
    return {'alphas': alpha.tolist(), 'alpha_bars': cumulative.tolist(),
            'scalar_clean': clean, 'scalar_epsilon': epsilon, 'scalar_noisy': noisy,
            'predicted_epsilon': predicted, 'noise_mse': (predicted-epsilon)**2,
            'estimated_clean': estimate, 'conditional_variance_at_bar064': .36,
            'posterior_variance_example': posterior_variance,
            'posterior_mean_from_x0': mean_from_x0, 'posterior_mean_from_noise': mean_from_noise}


class TinyDenoiser(nn.Module):
    def __init__(self, steps=32):
        super().__init__()
        self.time = nn.Embedding(steps+1, 8)
        self.layers = nn.Sequential(nn.Linear(16+8, 64), nn.SiLU(), nn.Linear(64, 16))

    def forward(self, noisy, step):
        # 时间编号先查成 8 维向量，再和 16 个图像坐标一起送入网络。
        return self.layers(torch.cat([noisy, self.time(step)], dim=-1))


def parameter_digest(model):
    hasher = hashlib.sha256()
    for parameter in model.parameters():
        hasher.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def train_diffusion(updates=2400):
    torch.manual_seed(17)
    steps = 32
    beta = torch.cat([torch.zeros(1), torch.linspace(.02, .2, steps)])
    alpha = 1-beta
    bar = torch.cumprod(alpha, dim=0)
    pattern = torch.tensor([[-1., 1., -1., 1.]]*4).reshape(16)
    patterns = torch.stack([pattern, -pattern])
    model = TinyDenoiser(steps)
    untrained = deepcopy(model)
    generator = torch.Generator().manual_seed(71)
    optimizer = torch.optim.Adam(model.parameters(), lr=.005)
    def batch(size, random):
        clean = patterns[torch.randint(2, (size,), generator=random)]
        time = torch.randint(1, steps+1, (size,), generator=random)
        noise = torch.randn(size, 16, generator=random)
        scale = bar[time, None]
        noisy = scale.sqrt()*clean+(1-scale).sqrt()*noise
        return noisy, time, noise
    eval_input, eval_time, eval_noise = batch(256, torch.Generator().manual_seed(117))
    with torch.no_grad():
        initial = (model(eval_input, eval_time)-eval_noise).square().mean().item()
    losses = []
    for iteration in range(updates):
        noisy, time, noise = batch(64, generator)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(noisy, time)-noise).square().mean()
        loss.backward(); optimizer.step()
        if iteration in [0, 99, 299, 599, 1199, updates-1]:
            losses.append({'update': iteration+1, 'training_loss': loss.item()})
    model.eval()
    digest_before_sampling = parameter_digest(model)
    with torch.no_grad():
        final = (model(eval_input, eval_time)-eval_noise).square().mean().item()
    def sample(network, clip_clean=True):
        random = torch.Generator().manual_seed(717)
        state = torch.randn(1, 16, generator=random)
        stages = [{'step': steps, 'image': state.reshape(4, 4).tolist()}]
        with torch.no_grad():
            for t in range(steps, 0, -1):
                noise = network(state, torch.tensor([t]))
                estimate = (state-(1-bar[t]).sqrt()*noise)/bar[t].sqrt()
                if clip_clean:
                    # 训练图样每个坐标落在 [-1,1]；按相同范围约束清晰图估计。
                    estimate = estimate.clamp(-1, 1)
                mean = (bar[t-1].sqrt()*beta[t]/(1-bar[t])*estimate+
                        alpha[t].sqrt()*(1-bar[t-1])/(1-bar[t])*state)
                variance = beta[t]*(1-bar[t-1])/(1-bar[t])
                state = mean
                if t > 1:
                    state = state+variance.sqrt()*torch.randn(state.shape, generator=random)
                if t-1 in [24, 16, 8, 0]:
                    stages.append({'step': t-1, 'image': state.reshape(4, 4).tolist()})
        distance = (state-patterns).square().mean(-1)
        return {'clip_clean_estimate': clip_clean, 'stages': stages, 'final': state.reshape(4, 4).tolist(),
                'mse_to_each_training_pattern': distance.tolist()}
    generated, random_generated = sample(model), sample(untrained)
    unclipped_generated = sample(model, clip_clean=False)
    assert final < initial
    assert parameter_digest(model) == digest_before_sampling
    return {'seed': 17, 'train_seed': 71, 'sampling_seed': 717, 'updates': updates,
            'batch_size': 64, 'diffusion_steps': steps, 'beta_first_last': [beta[1].item(), beta[-1].item()],
            'bar_alpha_T': bar[-1].item(), 'patterns': patterns.reshape(2, 4, 4).tolist(),
            'validation_initial_mse': initial, 'validation_final_mse': final,
            'training_observations': losses, 'generated': generated, 'untrained_generated': random_generated,
            'unclipped_generated': unclipped_generated,
            'parameters_unchanged_during_sampling': True, 'parameter_count': sum(p.numel() for p in model.parameters()),
            'weights_saved': False}
