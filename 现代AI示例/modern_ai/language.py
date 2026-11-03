"""小语言模型、LoRA、Patch 与配对表示实验。"""
import io
import hashlib
from time import perf_counter
import torch
from torch import nn
from .networks import CausalAttention, cache_experiment
from .mathlab import clip_loss


class TinyLM(nn.Module):
    def __init__(self, width=32, context=24):
        super().__init__()
        self.context = context
        # 字节词表与位置表分别描述内容和位置。
        self.embedding, self.position = nn.Embedding(256, width), nn.Embedding(context, width)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = CausalAttention(width, 2)
        self.ff = nn.Sequential(nn.Linear(width, width*2), nn.GELU(), nn.Linear(width*2, width))
        self.head = nn.Linear(width, 256)

    def forward(self, tokens):
        # 输入是 (batch,time) 的整型索引，输出是 (batch,time,256) 分数。
        length = tokens.shape[1]
        if length > self.context:
            raise ValueError('输入超出位置表')
        hidden = self.embedding(tokens) + self.position(torch.arange(length, device=tokens.device))
        attended, _, _ = self.attention(self.norm1(hidden))
        hidden = hidden + attended
        hidden = hidden + self.ff(self.norm2(hidden))
        return self.head(hidden)

    def forward_cached(self, tokens, past=None):
        # 新块的位置从历史缓存长度起算；超过位置表时由调用者重建窗口。
        old = 0 if past is None else past[0].shape[-2]
        length = tokens.shape[1]
        if old + length > self.context:
            raise ValueError('缓存前缀与新块超出位置表，需要重算窗口')
        positions = torch.arange(old, old + length, device=tokens.device)
        hidden = self.embedding(tokens) + self.position(positions)
        attended, cache, weights = self.attention(self.norm1(hidden), past)
        hidden = hidden + attended
        hidden = hidden + self.ff(self.norm2(hidden))
        return self.head(hidden), cache, weights


def evaluate_language(model, text):
    # 每个相邻字节目标只计一次；长前缀与生成时一样保留最后 context 项。
    values = torch.tensor(list(text), dtype=torch.long)
    total, count, mismatches, first_mismatch = 0., len(values)-1, 0, None
    model.eval()
    with torch.no_grad():
        for index in range(1, len(values)):
            prefix = values[max(0, index-model.context):index][None]
            scores = model(prefix)[:, -1]
            total += nn.functional.cross_entropy(scores, values[index:index+1]).item()
            predicted = int(scores.argmax(-1).item())
            if predicted != int(values[index]):
                mismatches += 1
                if first_mismatch is None:
                    first_mismatch = {'position': index, 'expected': int(values[index]),
                                      'predicted': predicted}
    return {'loss': total/count, 'targets': count, 'mismatches': mismatches,
            'first_mismatch': first_mismatch}


def train_language(return_model=False, *, training=None, validation=None):
    # 自编语料，无需下载；验证文本使用不同的词语组合。
    torch.manual_seed(17)
    torch.set_num_threads(2)
    training = (b'read notes.\nwrite plan.\nlearn math.\ncheck code.\n'*16
                if training is None else training)
    validation = (b'read plan.\nwrite notes.\ncheck math.\n'
                  if validation is None else validation)
    data = torch.tensor(list(training), dtype=torch.long)
    model, losses, started = TinyLM(), [], perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.008)
    for _ in range(140):
        # 同一窗口中的目标比输入向右偏移一个字节。
        starts = torch.randint(len(data)-model.context-1, (8,))
        batch = torch.stack([data[i:i+model.context] for i in starts])
        targets = torch.stack([data[i+1:i+model.context+1] for i in starts])
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = nn.functional.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    model.eval()
    validation_result = evaluate_language(model, validation)
    with torch.no_grad():
        tokens = torch.tensor([list(b'read ')], dtype=torch.long)
        for _ in range(24):
            # 贪心便于复现；采样入口可改为 multinomial。
            next_id = model(tokens[:, -model.context:])[:, -1].argmax(-1, keepdim=True)
            tokens = torch.cat((tokens, next_id), dim=1)
    result = {'seed': 17, 'device': 'cpu', 'steps': 140,
            'parameters': sum(p.numel() for p in model.parameters()),
            'initial_loss': losses[0], 'final_loss': losses[-1],
            'validation_loss': validation_result['loss'],
            'validation_targets': validation_result['targets'],
            'validation_mismatches': validation_result['mismatches'],
            'validation_first_mismatch': validation_result['first_mismatch'],
            'training_sha256': hashlib.sha256(training).hexdigest(),
            'validation_sha256': hashlib.sha256(validation).hexdigest(),
            'generated': bytes(tokens[0].tolist()).decode('utf-8', errors='replace'),
            'elapsed_ms': (perf_counter()-started)*1000}
    return (model, result) if return_model else result


def combination_experiment():
    # 两种训练配置共用一个保留集；其中四种完整组合均未出现在两份训练语料中。
    heldout = b'read code.\nwrite math.\nlearn notes.\ncheck plan.\n'
    original = b'read notes.\nwrite plan.\nlearn math.\ncheck code.\n'
    augmented = original+b'read plan.\nwrite notes.\ncheck math.\n'
    results = {}
    for name, corpus in [('original', original), ('augmented', augmented)]:
        results[name] = train_language(training=corpus*16, validation=heldout)
    return {'heldout': heldout.decode('ascii'), 'configurations': results}


class LoRALinear(nn.Module):
    def __init__(self, base, rank=2, alpha=2):
        super().__init__()
        self.base, self.scale = base, alpha/rank
        # 冻结基础层，训练两个低秩矩阵；B 零初始化使初始增量为零。
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.a = nn.Parameter(torch.randn(rank, base.in_features)*.01)
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x):
        # 两次小矩阵乘法的输出维度与基础层一致。
        return self.base(x) + (x @ self.a.T @ self.b.T)*self.scale


def lora_roundtrip():
    torch.manual_seed(17)
    layer = LoRALinear(nn.Linear(4, 3, bias=False))
    old = layer.base.weight.detach().clone()
    x, target = torch.randn(6, 4), torch.randn(6, 3)
    optimizer = torch.optim.SGD([layer.a, layer.b], lr=.1)
    optimizer.zero_grad(set_to_none=True)
    (layer(x)-target).square().mean().backward()
    # B 初值为零，第一次反向时 A 的梯度为零，B 已能接收非零梯度。
    gradient_norms = {'a': float(layer.a.grad.norm()), 'b': float(layer.b.grad.norm())}
    optimizer.step()
    # 内存流验证保存和加载，不制造训练权重副本。
    stream = io.BytesIO()
    torch.save(layer.state_dict(), stream)
    stream.seek(0)
    restored = LoRALinear(nn.Linear(4, 3, bias=False))
    restored.load_state_dict(torch.load(stream, weights_only=True))
    return {'base_unchanged': torch.equal(old, layer.base.weight),
            'trainable': sum(p.numel() for p in layer.parameters() if p.requires_grad),
            'first_gradient_norms': gradient_norms,
            'reload_error': float((layer(x)-restored(x)).abs().max().detach())}


def multimodal():
    torch.manual_seed(17)
    # 按左上、右上、左下、右下顺序抽取 2x2 Patch。
    image = torch.arange(16.).reshape(1, 1, 4, 4)
    patches = image.unfold(2, 2, 2).unfold(3, 2, 2)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(1, 4, 4)
    vision = nn.Linear(4, 6)(patches) + nn.Embedding(4, 6)(torch.arange(4))
    projected = nn.Linear(6, 8)(vision)
    text = nn.Embedding(256, 8)(torch.tensor([list(b'Q:')]))
    combined = torch.cat((projected, text), dim=1)
    # 双编码器学习两对图文的对应关系，完整输出配对与错配结果。
    image_features = torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.]])
    text_features = torch.eye(2)
    image_encoder, text_encoder = nn.Linear(4, 2), nn.Linear(2, 2)
    optimizer = torch.optim.Adam(list(image_encoder.parameters())+list(text_encoder.parameters()), lr=.03)
    values = []
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        i = nn.functional.normalize(image_encoder(image_features), dim=-1)
        t = nn.functional.normalize(text_encoder(text_features), dim=-1)
        scores = i @ t.T / .2
        labels = torch.arange(2)
        loss = (nn.functional.cross_entropy(scores, labels)+
                nn.functional.cross_entropy(scores.T, labels))/2
        values.append(loss.item())
        loss.backward()
        optimizer.step()
    return {'patches': patches.tolist(), 'vision_shape': list(vision.shape),
            'projected_shape': list(projected.shape), 'input_shape': list(combined.shape),
            'clip_hand': clip_loss([[1, 0], [0, 1]], [[1, 0], [0, 1]]),
            'alignment_initial': values[0], 'alignment_final': values[-1],
            'paired_predictions': scores.argmax(-1).tolist(),
            'swapped_text_predictions': scores[:, [1, 0]].argmax(-1).tolist()}


def demo():
    return {'tokenizer': {'text': '苹果 AI', 'utf8_ids': list('苹果 AI'.encode('utf-8'))},
            'language': train_language(), 'cache': cache_experiment(),
            'lora': lora_roundtrip(), 'multimodal': multimodal()}
