"""新第六篇：逐项像素/特征、损失、空间坐标与音频时间轴。"""
import argparse
import itertools
import json
import math
import sys
import numpy as np
import torch
from torch import nn
from .mathlab import softmax


def patch_trace():
    # arange 的编号用于追踪原像素，而不是把它们当作词表 ID。
    image = torch.arange(16., dtype=torch.float64).reshape(1, 1, 4, 4)
    blocks = image.unfold(2, 2, 2).unfold(3, 2, 2)
    ordered = blocks.permute(0, 2, 3, 1, 4, 5)
    patches = ordered.reshape(1, 4, 4)
    expected = [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]
    assert patches[0].tolist() == expected
    weight = torch.cat([torch.eye(4, dtype=torch.float64),
                        torch.full((4, 1), .25, dtype=torch.float64),
                        torch.tensor([[1.], [-1.], [1.], [-1.]], dtype=torch.float64)], dim=1)
    features = patches @ weight
    # 同一权重映射成卷积核；没有重新随机选一套模型。
    convolution = nn.Conv2d(1, 6, kernel_size=2, stride=2, bias=False).double()
    with torch.no_grad():
        convolution.weight.copy_(weight.T.reshape(6, 1, 2, 2))
    conv_features = convolution(image).flatten(2).transpose(1, 2)
    error = (features-conv_features).abs().max().item()
    assert error == 0
    # 以明确的逆轴排列把块放回去，验证没有丢像素或重复像素。
    restored = patches.reshape(1, 2, 2, 1, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(1, 1, 4, 4)
    assert torch.equal(image, restored)
    return {'image': image[0, 0].tolist(), 'unfold_shape': list(blocks.shape),
            'ordered_shape': list(ordered.shape), 'patches': patches[0].tolist(),
            'wrong_reshape': image.reshape(1, 4, 4)[0].tolist(),
            'projection_weight': weight.tolist(), 'features': features[0].tolist(),
            'conv_error': error, 'reconstructed_equal': True,
            'patch_positions': 4, 'with_cls_positions': 5}


def clip_trace():
    # 非对称打分使按行与按列归一化产生不同中间值。
    raw_u = np.array([[2., 0.], [0., 3.]])
    raw_v = np.array([[4., 0.], [3., 4.]])
    u = raw_u/np.linalg.norm(raw_u, axis=1, keepdims=True)
    v = raw_v/np.linalg.norm(raw_v, axis=1, keepdims=True)
    scores = u @ v.T
    rows, columns = softmax(scores, 1), softmax(scores, 0)
    image_loss = -np.log(np.diag(rows)).mean()
    text_loss = -np.log(np.diag(columns)).mean()
    gradient = (rows+columns-2*np.eye(2))/4
    gu, gv = gradient@v, gradient.T@u
    raw_gu = np.stack([(np.eye(2)-np.outer(a, a))@g/np.linalg.norm(x)
                       for a, g, x in zip(u, gu, raw_u)])
    raw_gv = np.stack([(np.eye(2)-np.outer(a, a))@g/np.linalg.norm(x)
                       for a, g, x in zip(v, gv, raw_v)])
    a = torch.tensor(raw_u, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(raw_v, dtype=torch.float64, requires_grad=True)
    normalized_a = nn.functional.normalize(a, dim=-1)
    normalized_b = nn.functional.normalize(b, dim=-1)
    logits = normalized_a@normalized_b.T
    loss = (nn.functional.cross_entropy(logits, torch.arange(2))+
            nn.functional.cross_entropy(logits.T, torch.arange(2)))/2
    loss.backward()
    np.testing.assert_allclose(raw_gu, a.grad.numpy(), atol=1e-14)
    np.testing.assert_allclose(raw_gv, b.grad.numpy(), atol=1e-14)
    # 中心差分独立检查原始特征，不以同一反传公式自证。
    def objective(value):
        unit = value/np.linalg.norm(value, axis=1, keepdims=True)
        s = unit@v.T
        return -(np.log(np.diag(softmax(s, 1))).mean()+
                 np.log(np.diag(softmax(s, 0))).mean())/2
    finite = np.empty_like(raw_u)
    for index in np.ndindex(raw_u.shape):
        plus, minus = raw_u.copy(), raw_u.copy()
        plus[index] += 1e-6; minus[index] -= 1e-6
        finite[index] = (objective(plus)-objective(minus))/2e-6
    np.testing.assert_allclose(finite, raw_gu, atol=1e-9)
    return {'u': u.tolist(), 'v': v.tolist(), 'scores': scores.tolist(),
            'row_probabilities': rows.tolist(), 'column_probabilities': columns.tolist(),
            'image_to_text_loss': float(image_loss), 'text_to_image_loss': float(text_loss),
            'loss': float(loss.detach()), 'score_gradient': gradient.tolist(),
            'unit_u_gradient': gu.tolist(), 'raw_u_gradient': raw_gu.tolist(),
            'raw_v_gradient': raw_gv.tolist(),
            'finite_difference_error': float(abs(finite-raw_gu).max()),
            'orthogonal_identity_loss': float(np.log1p(np.exp(-1)))}


def train_pairing():
    torch.manual_seed(17)
    image = torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.]])
    text = torch.eye(2)
    image_encoder, text_encoder = nn.Linear(4, 2), nn.Linear(2, 2)
    optimizer = torch.optim.Adam(list(image_encoder.parameters())+list(text_encoder.parameters()), lr=.03)
    def scores(inputs=image):
        a = nn.functional.normalize(image_encoder(inputs), dim=-1)
        b = nn.functional.normalize(text_encoder(text), dim=-1)
        return a@b.T/.2
    def objective(s):
        labels = torch.arange(2)
        return (nn.functional.cross_entropy(s, labels)+nn.functional.cross_entropy(s.T, labels))/2
    initial = objective(scores()).item()
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(scores())
        loss.backward(); optimizer.step()
    with torch.no_grad():
        final_scores = scores()
        heldout = scores(torch.tensor([[.9, .1, .8, .2], [.1, .9, .2, .8], [.5, .5, .5, .5]]))
    final = objective(final_scores).item()
    assert final < initial
    return {'seed': 17, 'steps': 100, 'temperature': .2, 'initial_loss': initial,
            'final_loss': final, 'scores': final_scores.tolist(),
            'predictions': final_scores.argmax(-1).tolist(),
            'swapped_predictions': final_scores[:, [1, 0]].argmax(-1).tolist(),
            'swapped_truth': [1, 0], 'heldout_scores': heldout.tolist(),
            'heldout_predictions': heldout.argmax(-1).tolist(),
            'ambiguous_input': [.5, .5, .5, .5], 'weights_saved': False}


def projector_trace():
    vision = torch.tensor(patch_trace()['features'], dtype=torch.float64)[None]
    right = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.], [0., 0.], [0., 0.]], dtype=torch.float64)
    projection = torch.cat([torch.eye(6, dtype=torch.float64), right], dim=1).requires_grad_()
    text = torch.zeros(1, 2, 8, dtype=torch.float64)
    text[0, 0, 0], text[0, 1, 1] = 1, 1
    wq, wk, wv = [torch.zeros(8, 2, dtype=torch.float64) for _ in range(3)]
    wq[1, 0], wk[4, 0], wk[5, 1] = 1, .1, .1
    wv[0, 0], wv[1, 1] = 1/15, 1/15
    head = torch.tensor([[1., -1., 0.], [0., 1., -1.]], dtype=torch.float64)
    def forward():
        projected = vision@projection
        joint = torch.cat([projected, text], dim=1)
        q, k, v = joint@wq, joint@wk, joint@wv
        raw_scores = q@k.transpose(-2, -1)/math.sqrt(2)
        visible = torch.ones(6, 6, dtype=torch.bool).tril()
        weights = raw_scores.masked_fill(~visible, -torch.inf).softmax(-1)
        output = weights@v
        logits = output[:, -1]@head
        loss = nn.functional.cross_entropy(logits, torch.tensor([1]))
        return projected, joint, weights, output, logits, loss
    projected, joint, weights, output, logits, loss = forward()
    loss.backward()
    old_loss, gradient_norm = loss.item(), projection.grad.norm().item()
    with torch.no_grad():
        projection -= .05*projection.grad
        new_loss = forward()[-1].item()
    assert gradient_norm > 0 and new_loss < old_loss
    return {'vision_shape': list(vision.shape), 'projected_shape': list(projected.shape),
            'text_shape': list(text.shape), 'joint_shape': list(joint.shape),
            'first_projected': projected[0, 0].tolist(),
            'last_query_weights': weights[0, -1].tolist(), 'last_readout': output[0, -1].tolist(),
            'logits': logits[0].tolist(), 'initial_loss': old_loss, 'updated_loss': new_loss,
            'projection_gradient_norm': gradient_norm,
            'tokens_p16_224': 196, 'tokens_p16_448': 784,
            'attention_with_64_text': [(196+64)**2, (784+64)**2]}


def box_iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not (a[2] > a[0] and a[3] > a[1] and b[2] > b[0] and b[3] > b[1]):
        raise ValueError('边界框必须有正面积')
    sides = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]))
    intersection = float(sides.prod())
    areas = float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2]))
    return intersection/(areas-intersection)


def spatial_trace():
    box = np.array([20., 10., 60., 50.])
    scaled = box*np.array([2., .5, 2., .5])
    flipped = np.array([200-scaled[2], scaled[1], 200-scaled[0], scaled[3]])
    mask = np.zeros((4, 4), dtype=int)
    mask[1:3, 1:3] = 1
    cropped = mask[:, 1:4]
    enlarged = np.repeat(np.repeat(cropped, 2, axis=0), 2, axis=1)
    final = enlarged[:, ::-1]
    positions = np.argwhere(final)
    recovered = [int(positions[:, 1].min()), int(positions[:, 0].min()),
                 int(positions[:, 1].max()+1), int(positions[:, 0].max()+1)]
    assert recovered == [2, 2, 6, 6]
    # 两个实例的语义类别相同，但像素身份标签不同。
    instances = np.zeros((4, 6), dtype=int)
    instances[1:3, :2], instances[1:3, 4:6] = 1, 2
    semantic = (instances > 0).astype(int)
    return {'pixel_box': box.tolist(), 'normalized_w100_h80': (box/[100, 80, 100, 80]).tolist(),
            'scaled_w200_h40': scaled.tolist(), 'flipped_w200': flipped.tolist(),
            'overlap_a': box.tolist(), 'overlap_b': [40, 30, 80, 70],
            'intersection': 400, 'union': 2800, 'iou': box_iou(box, [40, 30, 80, 70]),
            'mask_original': mask.tolist(), 'mask_crop': cropped.tolist(),
            'mask_resize_flip': final.tolist(), 'transformed_box': recovered,
            'instances': instances.tolist(), 'semantic': semantic.tolist()}


def train_segmentation():
    torch.manual_seed(17)
    target = torch.tensor(spatial_trace()['semantic'], dtype=torch.long)[None]
    image = target[:, None].float()
    model = nn.Conv2d(1, 2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.5)
    initial = nn.functional.cross_entropy(model(image), target).item()
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(image), target)
        loss.backward(); optimizer.step()
    with torch.no_grad():
        logits = model(image)
        predictions = logits.argmax(1)
        intersection = ((predictions == 1)&(target == 1)).sum().item()
        union = ((predictions == 1)|(target == 1)).sum().item()
    assert torch.equal(predictions, target)
    return {'steps': 120, 'logits_shape': list(logits.shape), 'target_shape': list(target.shape),
            'initial_loss': initial, 'final_loss': nn.functional.cross_entropy(logits, target).item(),
            'predictions': predictions[0].tolist(), 'foreground_iou': intersection/union,
            'foreground_pixel_probabilities': logits.softmax(1)[0, :, 1, 0].tolist(),
            'weights_saved': False}


def mel_filters(sr=16000, n_fft=512, bands=40):
    # 本书明确选 HTK 刻度、峰值 1 的三角窗，不隐含采用 Slaney 面积归一化。
    upper = 2595*np.log10(1+(sr/2)/700)
    mel_points = np.linspace(0, upper, bands+2)
    hz_points = 700*(10**(mel_points/2595)-1)
    bins = np.arange(n_fft//2+1)*sr/n_fft
    bank = np.zeros((bands, len(bins)))
    for row in range(bands):
        left, center, right = hz_points[row:row+3]
        bank[row] = np.maximum(0, np.minimum((bins-left)/(center-left), (right-bins)/(right-center)))
    return bank, hz_points


def audio_trace():
    sr, samples = 16000, 32000
    time = np.arange(samples)/sr
    # 前一秒基频 200Hz，后一秒 300Hz；在整周期边界连续交接相位。
    phase = np.where(time < 1, 2*np.pi*200*time, 2*np.pi*(200+300*(time-1)))
    wave = .6*np.sin(phase)+.2*np.sin(2*phase)
    length, hop, n_fft = 400, 160, 512
    frames = np.lib.stride_tricks.sliding_window_view(wave, length)[::hop]
    window = .5-.5*np.cos(2*np.pi*np.arange(length)/length)
    spectrum = np.fft.rfft(frames*window, n=n_fft, axis=-1)
    power = abs(spectrum)**2/n_fft
    bank, hz_points = mel_filters(sr, n_fft, 40)
    mel = bank@power.T
    logged = np.log(np.maximum(mel, 1e-10))
    # 手算帧恰含一周期：16 点、1000Hz；频点 1 为 -8i。
    hand = np.sin(2*np.pi*np.arange(16)/16)
    hand_fft = np.fft.rfft(hand)
    np.testing.assert_allclose(hand_fft[1], -8j, atol=1e-13)
    # 在明确的周期范围中找合成波的自相关峰。
    short = wave[:1600]-wave[:1600].mean()
    lags = np.arange(54, 101)
    correlations = np.array([np.dot(short[:-lag], short[lag:]) /
                             np.sqrt(np.dot(short[:-lag], short[:-lag])*np.dot(short[lag:], short[lag:]))
                             for lag in lags])
    best_lag = int(lags[np.argmax(correlations)])
    assert best_lag == 80
    # 明确展示 6kHz 与 2kHz 在 8kHz 采样下的混叠。
    n = np.arange(32)
    alias_error = float(abs(np.sin(2*np.pi*6000*n/8000)+np.sin(2*np.pi*2000*n/8000)).max())
    return {'sample_rate': sr, 'samples': samples, 'duration_seconds': samples/sr,
            'waveform_first160': wave[:160].tolist(), 'first_frame_first8': frames[0, :8].tolist(),
            'window_samples': length, 'hop_samples': hop, 'fft_size': n_fft,
            'frame_shape': list(frames.shape), 'power_shape': list(power.shape),
            'mel_shape': list(logged.shape),
            'mel_display_indices': np.linspace(0, len(frames)-1, 32).astype(int).tolist(),
            'mel_display': logged[:, np.linspace(0, len(frames)-1, 32).astype(int)].tolist(),
            'fundamental_segments_hz': [200, 300],
            'mel_hz_points_first8': hz_points[:8].tolist(),
            'power_first_frame': power[0].tolist(), 'tail_unused_samples': samples-((len(frames)-1)*hop+length),
            'hand_wave': hand.tolist(), 'hand_fft_real': hand_fft.real.tolist(),
            'hand_fft_imag': hand_fft.imag.tolist(), 'best_lag': best_lag,
            'estimated_f0_hz': sr/best_lag, 'f0_after_12_semitones': 400.,
            'alias_error': alias_error, 'stride2_encoder_frames': (len(frames)+1)//2}


def collapse_ctc(path, blank=0):
    # 必须先合并相邻重复，再删除 blank；交换次序会吞掉合法的重复字。
    merged = [v for i, v in enumerate(path) if i == 0 or v != path[i-1]]
    return [v for v in merged if v != blank]


def ctc_trace():
    probabilities = np.array([[.2, .7, .1], [.6, .3, .1], [.2, .2, .6]])
    target = [1]
    valid = []
    total = 0.
    for path in itertools.product(range(3), repeat=3):
        if collapse_ctc(path) == target:
            probability = float(np.prod(probabilities[np.arange(3), path]))
            total += probability
            valid.append({'path': list(path), 'probability': probability})
    # 扩展状态为 blank,a,blank；最后两状态的和就是目标总概率。
    alpha = np.zeros((3, 3))
    alpha[0, :2] = probabilities[0, :2]
    states = [0, 1, 0]
    for t in range(1, 3):
        for state, symbol in enumerate(states):
            previous = alpha[t-1, state]
            if state > 0:
                previous += alpha[t-1, state-1]
            alpha[t, state] = previous*probabilities[t, symbol]
    np.testing.assert_allclose(alpha[-1, -2:].sum(), total)
    z = torch.tensor(np.log(probabilities), dtype=torch.float64, requires_grad=True)
    loss = nn.CTCLoss(blank=0, reduction='sum')(z.log_softmax(-1)[:, None], torch.tensor(target),
                                               torch.tensor([3]), torch.tensor([1]))
    loss.backward()
    np.testing.assert_allclose(loss.item(), -np.log(total), atol=1e-13)
    return {'probabilities': probabilities.tolist(), 'valid_paths': valid, 'target_probability': total,
            'alpha': alpha.tolist(), 'loss': loss.item(), 'logit_gradient': z.grad.tolist(),
            'greedy_path': probabilities.argmax(-1).tolist(),
            'greedy_result': collapse_ctc(probabilities.argmax(-1).tolist()),
            'repeated_symbol_path': [0, 1, 1, 0, 1, 2, 2, 0],
            'repeated_symbol_result': collapse_ctc([0, 1, 1, 0, 1, 2, 2, 0])}


def retrieve_features(query, bank, k=2, rate=.75):
    query, bank = np.asarray(query, float), np.asarray(bank, float)
    if query.shape[-1] != bank.shape[-1] or not 1 <= k <= len(bank) or not 0 <= rate <= 1:
        raise ValueError('维度、邻居数或混合比例不合法')
    distances = ((bank-query)**2).sum(axis=1)
    indices = np.argsort(distances, kind='stable')[:k]
    selected = distances[indices]
    # 精确零距离单独分配权重，避免 1/0 和 inf/inf。
    if np.any(selected == 0):
        weights = (selected == 0).astype(float)
    else:
        weights = (1/selected)**2
    weights /= weights.sum()
    retrieved = weights@bank[indices]
    mixed = rate*retrieved+(1-rate)*query
    return {'indices': indices.tolist(), 'squared_distances': selected.tolist(),
            'weights': weights.tolist(), 'retrieved': retrieved.tolist(), 'mixed': mixed.tolist()}


def rvc_trace():
    bank = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [2., 0.], [0., 2.], [-1., 0.], [0., -1.]])
    result = retrieve_features([.2, 0.], bank[:2], 2, .75)
    np.testing.assert_allclose(result['weights'], [256/257, 1/257])
    exact = retrieve_features([0., 0.], bank, 8, .75)
    assert exact['retrieved'] == [0., 0.]
    # 与该版 F.interpolate 默认 nearest、scale_factor=2 一样，逐帧重复。
    content = torch.tensor([[[.1, .2], [.3, .4], [.5, .6]]])
    expanded = nn.functional.interpolate(content.transpose(1, 2), scale_factor=2).transpose(1, 2)
    raw_pitch = np.array([200., 0., 220., 0., 200., 0.])
    voiced = raw_pitch > 0
    interpolated = raw_pitch.copy()
    interpolated[~voiced] = np.interp(np.where(~voiced)[0], np.where(voiced)[0], raw_pitch[voiced])
    shifted = interpolated*2
    return {'bank': bank.tolist(), 'query': [.2, 0.], 'hand_k2': result,
            'eight_neighbor_result': retrieve_features([.2, 0.], bank, 8, .75),
            'exact_match': exact, 'content_shape': list(content.shape),
            'expanded_shape': list(expanded.shape), 'expanded_content': expanded.tolist(),
            'raw_f0': raw_pitch.tolist(), 'original_voiced_mask': voiced.tolist(),
            'interpolated_f0': interpolated.tolist(), 'shifted_12_semitones': shifted.tolist(),
            'source_pin': '81eed5e8f68b6bed1789f682fe78cdd324495afc',
            'official_pipeline_executed': False, 'faiss_package_executed': False}


def speech_chain():
    # 这一组是确定性的交接记录，用来定位错误；未调用 ASR/TTS 模型。
    original = '不要覆盖原文件'
    transcripts = [original, '覆盖原文件']
    duration = [3, 2]
    phonemes = np.array([[1., .2], [.1, 1.]])
    acoustic = np.repeat(phonemes, duration, axis=0)
    return {'utterance_intent': original, 'transcripts': transcripts,
            'actions': ['保留原文件并新建结果', '覆盖原文件'],
            'tts_phonemes': ['a', 'i'], 'durations_in_frames': duration,
            'expanded_features': acoustic.tolist(), 'frames': len(acoustic),
            'hop_samples': 160, 'nominal_samples': len(acoustic)*160,
            'asr_tts_models_executed': False}


def exercise(code):
    functions = {'MM-01': patch_trace, 'MM-02': clip_trace, 'MM-03': projector_trace,
                 'MM-04': spatial_trace, 'MM-06': audio_trace,
                 'MM-07': lambda: {'ctc': ctc_trace(), 'speech': speech_chain()}, 'MM-08': rvc_trace}
    if code == 'MM-05':
        from .diffusion_lab import diffusion_trace
        return diffusion_trace()
    return functions[code]()


def demo(train=False):
    torch.set_num_threads(2)
    result = {'patch': patch_trace(), 'clip': clip_trace(), 'projector': projector_trace(),
              'spatial': spatial_trace(), 'audio': audio_trace(), 'ctc': ctc_trace(),
              'speech': speech_chain(), 'rvc': rvc_trace()}
    from .diffusion_lab import diffusion_trace, train_diffusion
    result['diffusion'] = diffusion_trace()
    if train:
        result['pairing_training'] = train_pairing()
        result['segmentation_training'] = train_segmentation()
        result['diffusion_training'] = train_diffusion()
    return result


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    print(json.dumps(demo(args.train), ensure_ascii=False, indent=2))
