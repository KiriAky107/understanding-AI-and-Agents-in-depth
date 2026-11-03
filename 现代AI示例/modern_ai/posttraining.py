"""新第五篇：回答监督、偏好更新、PPO、低秩与量化的可运行计算。"""
from copy import deepcopy
import argparse
import hashlib
import io
import json
import math
import re
import struct
import sys
import zipfile

import numpy as np
import torch
from torch import nn
from .language import TinyLM, LoRALinear, train_language


def answer_batch(records):
    # 每项是字节提示与字节回答；尾部 padding 的目标忽略，不进入更早的因果查询。
    lengths = [len(prompt + answer) - 1 for prompt, answer in records]
    inputs = torch.zeros(len(records), max(lengths), dtype=torch.long)
    labels = torch.full_like(inputs, -100)
    for row, (prompt, answer) in enumerate(records):
        if not prompt or not answer:
            raise ValueError('提示与回答均须非空')
        sequence = torch.tensor(list(prompt + answer), dtype=torch.long)
        inputs[row, :lengths[row]] = sequence[:-1]
        # 预测首个回答字节的查询位是提示最后一项，而非回答首项。
        start = len(prompt) - 1
        labels[row, start:lengths[row]] = sequence[start+1:]
    return inputs, labels


def answer_loss(logits, labels):
    if not bool((labels != -100).any()):
        raise ValueError('没有有效回答目标')
    return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       labels.reshape(-1), ignore_index=-100)


def sequence_logprob(model, prompt, answer):
    # 求的是给定提示下整段回答的对数概率和；提示损失不进入这个和。
    inputs, labels = answer_batch([(prompt, answer)])
    logits = model(inputs)
    selected = labels != -100
    values = logits.log_softmax(-1)[selected]
    return values.gather(1, labels[selected][:, None]).sum()


def sft_gradient():
    torch.manual_seed(17)
    model = TinyLM().double()
    inputs, labels = answer_batch([(b'Q:a\nA:', b'b')])
    captured = []
    def save_activation(module, arguments, output):
        output.retain_grad()
        captured.append(output)
    handle = model.embedding.register_forward_hook(save_activation)
    try:
        logits = model(inputs)
        logits.retain_grad()
        loss = answer_loss(logits, labels)
        loss.backward()
        direct = logits.grad.norm(dim=-1)[0]
        activation = captured[0].grad.norm(dim=-1)[0]
    finally:
        handle.remove()
    assert torch.count_nonzero(direct[:-1]) == 0
    assert activation[0] > 0 and direct[-1] > 0
    rejected = False
    try:
        answer_loss(logits, torch.full_like(labels, -100))
    except ValueError:
        rejected = True
    assert rejected
    return {'inputs': inputs.tolist(), 'labels': labels.tolist(),
            'loss': loss.item(), 'valid_targets': int((labels != -100).sum()),
            'direct_logit_gradient_norms': direct.tolist(),
            'embedding_activation_gradient_norms': activation.tolist(),
            'empty_answer_rejected': rejected}


def dpo_math():
    # 三项分布中前两项是同一提示的胜出/落败回答，第三项是其他回答。
    logits = torch.tensor([.4, .1, .5], dtype=torch.float64).log().requires_grad_()
    reference = torch.tensor([.2, .2, .6], dtype=torch.float64).log()
    beta = .5
    logp = logits.log_softmax(-1)
    delta = beta * ((logp[0]-reference[0]) - (logp[1]-reference[1]))
    loss = -nn.functional.logsigmoid(delta)
    delta.retain_grad()
    loss.backward()
    gradient = logits.grad.detach().clone()
    with torch.no_grad():
        new_logits = logits - .5 * gradient
        new_logp = new_logits.log_softmax(-1)
        new_delta = beta*((new_logp[0]-reference[0])-(new_logp[1]-reference[1]))
        new_loss = -nn.functional.logsigmoid(new_delta)
    np.testing.assert_allclose(gradient.numpy(), [-1/6, 1/6, 0], atol=1e-14)
    assert new_loss < loss
    return {'beta': beta, 'reference_probabilities': reference.exp().tolist(),
            'probabilities': logp.exp().tolist(), 'delta': delta.item(),
            'preference_probability': delta.sigmoid().item(), 'loss': loss.item(),
            'd_delta': delta.grad.item(), 'logit_gradient': gradient.tolist(),
            'updated_probabilities': new_logp.exp().tolist(),
            'updated_loss': new_loss.item(), 'initial_equal_model_loss': math.log(2)}


def ppo_trace():
    rows = []
    for advantage in [1., -1.]:
        for value in [.6, 1., 1.5]:
            ratio = torch.tensor(value, dtype=torch.float64, requires_grad=True)
            term = torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage)
            term.backward()
            rows.append({'advantage': advantage, 'ratio': value,
                         'objective': term.item(), 'd_objective_d_ratio': ratio.grad.item()})
    rewards = np.array([0., 0., 1.])
    values = np.array([.4, .5, .6, 0.])
    gamma, lam = .9, .8
    returns = np.array([gamma**2, gamma, 1.])
    residuals = rewards + gamma*values[1:] - values[:-1]
    gae = np.zeros(3)
    tail = 0.
    for t in range(2, -1, -1):
        tail = residuals[t] + gamma*lam*tail
        gae[t] = tail
    p, q = np.array([.98, .02]), np.array([.99, .01])
    return {'epsilon': .2, 'terms': rows, 'gamma': gamma, 'lambda': lam,
            'rewards': rewards.tolist(), 'values': values.tolist(),
            'returns': returns.tolist(), 'advantages': (returns-values[:-1]).tolist(),
            'td_residuals': residuals.tolist(), 'gae': gae.tolist(),
            'small_kl_with_doubled_bad_probability': float((p*np.log(p/q)).sum()),
            'loose_verifier_accepts_3911': '391' in '3911',
            'strict_verifier_accepts_3911': bool(re.fullmatch(r'391', '3911'))}


def ppo_bandit():
    # 一步轨迹：动作 0 奖励 1、动作 1 奖励 0；确实抽样、反传并更新策略与价值。
    torch.manual_seed(17)
    logits = nn.Parameter(torch.zeros(2, dtype=torch.float64))
    value = nn.Parameter(torch.zeros((), dtype=torch.float64))
    actor = torch.optim.SGD([logits], lr=.4)
    critic = torch.optim.SGD([value], lr=.2)
    reference = torch.full((2,), .5, dtype=torch.float64)
    history = []
    for iteration in range(20):
        with torch.no_grad():
            old_logp = logits.log_softmax(-1).clone()
            actions = torch.multinomial(old_logp.exp(), 64, replacement=True)
            rewards = (actions == 0).double()
            advantages = (rewards - value).detach()
        for epoch in range(4):
            logp = logits.log_softmax(-1)
            ratios = (logp[actions] - old_logp[actions]).exp()
            clipped = torch.minimum(ratios*advantages, ratios.clamp(.8, 1.2)*advantages)
            kl = (logp.exp()*(logp-reference.log())).sum()
            actor.zero_grad(set_to_none=True)
            (-clipped.mean()+.02*kl).backward()
            actor.step()
            critic.zero_grad(set_to_none=True)
            ((value-rewards).square().mean()/2).backward()
            critic.step()
        history.append({'iteration': iteration, 'sampled_successes': int(rewards.sum()),
                        'probability_action0': logits.softmax(-1)[0].item(),
                        'value': value.item()})
    assert history[-1]['probability_action0'] > .8
    return {'seed': 17, 'batches': 20, 'batch_size': 64, 'epochs_per_batch': 4,
            'initial_probability_action0': .5, 'reference': reference.tolist(),
            'history': history, 'final_probabilities': logits.softmax(-1).tolist()}


def reasoning_style():
    # 两个回答风格类别：模板开场/直接表达。训练实际更新类别 logits。
    results = {}
    for name, counts in [('template_majority', [8., 2.]), ('plain_majority', [2., 8.])]:
        logits = nn.Parameter(torch.zeros(2, dtype=torch.float64))
        desired = torch.tensor(counts, dtype=torch.float64)/sum(counts)
        optimizer = torch.optim.SGD([logits], lr=.3)
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            loss = -(desired*logits.log_softmax(-1)).sum()
            loss.backward()
            optimizer.step()
        results[name] = logits.softmax(-1).tolist()
    preference = {}
    for chosen in [0, 1]:
        logits = nn.Parameter(torch.tensor([.8, .2], dtype=torch.float64).log())
        ref = logits.detach().clone().log_softmax(-1)
        optimizer = torch.optim.SGD([logits], lr=.1)
        for _ in range(40):
            logp = logits.log_softmax(-1)
            delta = (logp[chosen]-ref[chosen])-(logp[1-chosen]-ref[1-chosen])
            optimizer.zero_grad(set_to_none=True)
            (-nn.functional.logsigmoid(delta)).backward()
            optimizer.step()
        preference[str(chosen)] = logits.softmax(-1).tolist()
    correct = {'parts': [17*20, 17*3], 'sum': 17*20+17*3, 'valid': True}
    wrong = {'parts': [340, 41], 'sum': 381, 'valid': 41 == 17*3}
    return {'sft_style_probabilities': results, 'dpo_chosen_index_probabilities': preference,
            'low_temperature_template_majority': 16/17,
            'low_temperature_plain_majority': 1/17,
            'correct_scratchpad': correct, 'wrong_scratchpad': wrong,
            'direct_bytes': len(b'391'),
            'scratchpad_bytes': len(b'17*20=340;17*3=51;340+51=391')}


def lora_trace():
    # 与旧双路 SVG 相同的 4→2→3 数值；另用二维输入画秩一增量的像。
    w = np.eye(3, 4)
    a = np.array([[1., 0, 1, 0], [0, 1, 0, 1]])
    b = np.array([[1., 0], [0, 1], [1, -1]])
    x = np.arange(1., 5.)
    gradient_cases = []
    for name, a0, b0 in [('a_nonzero', [[1., 1.]], [[0.], [0.]]),
                         ('both_zero', [[0., 0.]], [[0.], [0.]]),
                         ('b_nonzero', [[0., 0.]], [[1.], [1.]])]:
        at = torch.tensor(a0, dtype=torch.float64, requires_grad=True)
        bt = torch.tensor(b0, dtype=torch.float64, requires_grad=True)
        xt = torch.tensor([1., 2.], dtype=torch.float64)
        output = xt + bt @ (at @ xt)
        loss = output.square().sum()/2
        loss.backward()
        ag, bg = at.grad.detach().clone(), bt.grad.detach().clone()
        with torch.no_grad():
            updated_a, updated_b = at-.01*ag, bt-.01*bg
            updated_output = xt+updated_b@(updated_a@xt)
            updated_loss = updated_output.square().sum()/2
        gradient_cases.append({'case': name, 'a_gradient': ag.tolist(), 'b_gradient': bg.tolist(),
                               'initial_loss': loss.item(), 'updated_loss': updated_loss.item(),
                               'updated_output': updated_output.tolist()})
    # 非零 A、零 B 的第二轮，A 才开始收到来自已改变 B 的梯度。
    at = torch.tensor([[1., 1.]], dtype=torch.float64, requires_grad=True)
    bt = torch.tensor([[-.03], [-.06]], dtype=torch.float64, requires_grad=True)
    xt = torch.tensor([1., 2.], dtype=torch.float64)
    ((xt+bt@(at@xt)).square().sum()/2).backward()
    delta = np.array([[2., 4.], [1., 2.]])
    merged = np.eye(2)+delta
    np.testing.assert_allclose(w@x+b@(a@x), [5., 8., 1.])
    assert np.linalg.matrix_rank(delta)==1 and np.linalg.matrix_rank(merged)==2
    return {'base': (w@x).tolist(), 'narrow': (a@x).tolist(),
            'update': (b@a@x).tolist(), 'output': (w@x+b@a@x).tolist(),
            'initialization_cases': gradient_cases, 'second_a_gradient': at.grad.tolist(),
            'rank_one_update': delta.tolist(), 'merged_rank': int(np.linalg.matrix_rank(merged)),
            'merged_determinant': float(np.linalg.det(merged)),
            'large_matrix_parameters': 1024**2, 'large_adapter_parameters': 8*(1024+1024)}


def quantization_trace():
    weights = np.array([-1., -.2, .3, 1.])
    x = np.array([1., 2., -1., .5])
    result = {'weights': weights.tolist(), 'input': x.tolist(), 'original_output': float(weights@x)}
    for bits, limit in [(8, 127), (4, 7)]:
        scale = float(abs(weights).max()/limit)
        q = np.clip(np.rint(weights/scale), -limit, limit).astype(np.int8)
        restored = q.astype(float)*scale
        error = restored-weights
        assert abs(error).max() <= scale/2+1e-14
        result[str(bits)] = {'scale': scale, 'integers': q.tolist(), 'restored': restored.tolist(),
                             'errors': error.tolist(), 'output': float(restored@x),
                             'output_error': float((restored-weights)@x),
                             'output_error_bound': float(abs(x).sum()*scale/2)}
    q4 = np.array(result['4']['integers'], dtype=np.int8)
    nibbles = q4.astype(np.int16) & 15
    packed = (nibbles[::2] | (nibbles[1::2] << 4)).astype(np.uint8)
    unpacked = np.empty(4, dtype=np.int16)
    unpacked[::2], unpacked[1::2] = packed & 15, packed >> 4
    unpacked = np.where(unpacked >= 8, unpacked-16, unpacked)
    np.testing.assert_array_equal(unpacked, q4)
    result['int4_packing'] = {'low_nibble_first': True, 'bytes': packed.tolist(),
                             'unpacked': unpacked.tolist(), 'payload_bytes': packed.nbytes,
                             'including_fp32_scale_bytes': packed.nbytes+4}
    result['clipping'] = {'input_weight': 1.8, 'fixed_scale_restored': 1., 'error': -.8}
    result['group128_fp16_scale_bytes_per_weight'] = .5+2/128
    return result


def parameter_ledger():
    model = TinyLM()
    groups = {'embedding_and_position': 256*32+24*32,
              'attention': 4*32*32, 'feedforward': 32*64+64+64*32+32,
              'normalizations': 2*(32+32), 'output': 32*256+256}
    assert sum(groups.values()) == sum(p.numel() for p in model.parameters()) == 25824
    body = 12*32*4096**2
    return {'tiny_groups': groups, 'tiny_total': sum(groups.values()),
            'standard_32_layer_body': body, 'v32000_untied_embeddings': 2*32000*4096,
            'standard_body_plus_untied': body+2*32000*4096,
            'seven_billion_fp32_bytes': 7_000_000_000*4,
            'seven_billion_fp16_bytes': 7_000_000_000*2,
            'seven_billion_int4_payload_bytes': 7_000_000_000//2,
            'seven_billion_group128_bytes': int(7_000_000_000*(.5+2/128))}


def serialization_trace():
    # 全部字节位于内存，不创建权重文件或解压目录。
    tensor = torch.tensor([[1., 2.], [3., 4.]])
    stream = io.BytesIO()
    torch.save({'W': tensor}, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=True)
    assert torch.equal(tensor, restored['W'])
    stream.seek(0)
    with zipfile.ZipFile(stream) as archive:
        names = archive.namelist()
    # 按公开布局组成一个单张量容器，演示偏移；不是替代官方实现的通用解析器。
    raw = tensor.numpy().astype('<f4').tobytes()
    header = json.dumps({'W': {'dtype': 'F32', 'shape': [2, 2], 'data_offsets': [0, 16]}},
                        separators=(',', ':')).encode()
    header += b' '*((-len(header)) % 8)
    payload = struct.pack('<Q', len(header))+header+raw
    size = struct.unpack('<Q', payload[:8])[0]
    metadata = json.loads(payload[8:8+size])
    offset = 8+size
    recovered = np.frombuffer(payload, dtype='<f4', offset=offset).reshape(2, 2)
    np.testing.assert_array_equal(recovered, tensor.numpy())
    return {'pt_memory_roundtrip_equal': True, 'pt_archive_entries': names,
            'safetensors_layout': {'header_bytes': size, 'data_start': offset,
                                   'data_bytes': len(raw), 'metadata': metadata,
                                   'layout_roundtrip_equal': True,
                                   'official_parser_executed': False},
            'files_created': False, 'onnx_export_executed': False, 'gguf_inference_executed': False}


def tiny_adaptation():
    # 同一基座只训练一次。后训练的三个分支在内存里比较，结束即释放。
    base, pretraining = train_language(return_model=True)
    records = [(b'Q:a\nA:', b'b'), (b'Q:c\nA:', b'd')]
    inputs, labels = answer_batch(records)
    held_inputs, held_labels = answer_batch([(b'Q:e\nA:', b'f')])
    model = deepcopy(base).train()
    initial = answer_loss(model(inputs), labels).item()
    held_initial = answer_loss(model(held_inputs), held_labels).item()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = answer_loss(model(inputs), labels)
        loss.backward()
        optimizer.step()
    final = answer_loss(model(inputs), labels).item()
    sft = {'steps': 60, 'initial_loss': initial, 'final_loss': final,
           'initial_heldout_loss': held_initial,
           'final_heldout_loss': answer_loss(model(held_inputs), held_labels).item(),
           'trained_next_bytes': model(inputs)[:, -1].argmax(-1).tolist(),
           'heldout_next_byte': int(model(held_inputs)[0, -1].argmax())}
    assert final < initial
    reference = deepcopy(model).requires_grad_(False)
    before_ref = {name: value.detach().clone() for name, value in reference.state_dict().items()}
    pairs = [(b'Q:a\nA:', b'b', b'c'), (b'Q:c\nA:', b'd', b'e')]
    def pair_scores():
        log_pairs = [(sequence_logprob(model, p, good), sequence_logprob(model, p, bad))
                     for p, good, bad in pairs]
        return [[float(a.detach()), float(b.detach())] for a, b in log_pairs]
    old_scores = pair_scores()
    with torch.no_grad():
        ref_pairs = [(sequence_logprob(reference, p, good), sequence_logprob(reference, p, bad))
                     for p, good, bad in pairs]
    optimizer = torch.optim.SGD(model.parameters(), lr=.005)
    dpo_losses = []
    for _ in range(30):
        terms = []
        for (p, good, bad), (ref_good, ref_bad) in zip(pairs, ref_pairs):
            delta = .5*((sequence_logprob(model, p, good)-ref_good)
                        -(sequence_logprob(model, p, bad)-ref_bad))
            terms.append(-nn.functional.logsigmoid(delta))
        loss = torch.stack(terms).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        dpo_losses.append(loss.item())
    new_scores = pair_scores()
    reference_unchanged = all(torch.equal(value, before_ref[name])
                              for name, value in reference.state_dict().items())
    assert reference_unchanged and dpo_losses[-1] < dpo_losses[0]
    # LoRA 仅修改输出层；所有原权重冻结，不存第二套项目或权重文件。
    adapted = deepcopy(base)
    adapted.requires_grad_(False)
    torch.manual_seed(17)
    adapted.head = LoRALinear(adapted.head, rank=2, alpha=2)
    frozen = {name: p.detach().clone() for name, p in adapted.named_parameters() if not p.requires_grad}
    trainable = [p for p in adapted.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=.05)
    lora_initial = answer_loss(adapted(inputs), labels).item()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = answer_loss(adapted(inputs), labels)
        loss.backward()
        optimizer.step()
    lora_final = answer_loss(adapted(inputs), labels).item()
    frozen_equal = all(torch.equal(p, frozen[name]) for name, p in adapted.named_parameters()
                       if name in frozen)
    with torch.no_grad():
        original = adapted(inputs)
        head = adapted.head
        head.base.weight.add_((head.b@head.a)*head.scale)
        adapted.head = head.base
        merge_error = (adapted(inputs)-original).abs().max().item()
    assert frozen_equal and lora_final < lora_initial and merge_error < 1e-4
    return {'pretraining': pretraining, 'records': [{'prompt': p.decode(), 'answer': a.decode()} for p, a in records],
            'heldout': {'prompt': 'Q:e\nA:', 'answer': 'f'}, 'sft': sft,
            'dpo': {'steps': 30, 'losses_first_last': [dpo_losses[0], dpo_losses[-1]],
                    'before_log_probabilities': old_scores, 'after_log_probabilities': new_scores,
                    'reference_unchanged': reference_unchanged},
            'lora': {'steps': 80, 'trainable_parameters': sum(p.numel() for p in trainable),
                     'initial_loss': lora_initial, 'final_loss': lora_final,
                     'frozen_parameters_unchanged_before_merge': frozen_equal,
                     'merge_max_error': merge_error}, 'weights_saved': False}


def exercise(code):
    functions = {'PT-01': sft_gradient, 'PT-02': dpo_math,
                 'PT-03': lambda: {'calculation': ppo_trace(), 'training': ppo_bandit()},
                 'PT-04': reasoning_style, 'PT-05': lora_trace,
                 'PT-06': lambda: {'quantization': quantization_trace(), 'parameters': parameter_ledger(),
                                   'serialization': serialization_trace()}}
    return functions[code]()


def demo(train=False):
    torch.set_num_threads(2)
    result = {'sft_gradient': sft_gradient(), 'dpo_math': dpo_math(),
              'ppo': ppo_trace(), 'ppo_training': ppo_bandit(),
              'reasoning_style': reasoning_style(), 'lora': lora_trace(),
              'quantization': quantization_trace(), 'parameters': parameter_ledger(),
              'serialization': serialization_trace()}
    if train:
        result['tiny_adaptation'] = tiny_adaptation()
    return result


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    print(json.dumps(demo(args.train), ensure_ascii=False, indent=2))
