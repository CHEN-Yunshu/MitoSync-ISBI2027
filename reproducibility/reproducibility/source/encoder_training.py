#!/usr/bin/env python3
"""Amendment 10 Part B: train the registered ResNet-18 division detector (one arm, one seed).

Registered: ImageNet ResNet-18, 3 channels = frames t-1,t,t+1, 48x48 crops resized to 96x96, AdamW lr 3e-4,
wd 1e-4, batch 256, 6000 steps, cosine, flips + 90-degree rotations, no early stopping, final checkpoint.
Sampling: R = 100 % real; synthetic arms 50/50 real/synthetic; positives:negatives 1:10 per source.
Open details: DESIGN_DETAILS.md section 4.
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARMS, REAL, OUT, MEAN, STD, load_source

STEPS, BATCH, LR, WD, SIZE = 6000, 256, 3e-4, 1e-4, 96


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S %Z')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['psc', 'fluo'], required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--steps', type=int, default=STEPS, help='only for smoke tests; registered value 6000')
    ap.add_argument('--out', default=str(OUT / 'runs'))
    a = ap.parse_args()
    assert a.arm in ARMS[a.dataset]
    name = f'{a.dataset}_{a.arm}_s{a.seed}' + ('' if a.steps == STEPS else f'_smoke{a.steps}')
    rd = Path(a.out) / name
    rd.mkdir(parents=True, exist_ok=True)
    assert not (rd / 'model.pt').exists(), f'{rd} already has a final model'
    logf = open(rd / 'train.log', 'a')

    def log(*x):
        s = ' '.join(str(v) for v in (now(),) + x)
        print(s, flush=True); logf.write(s + '\n'); logf.flush()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    torch.backends.cudnn.benchmark = True
    dev = torch.device('cuda')
    srcs = [REAL[a.dataset]] + ARMS[a.dataset][a.arm]
    per_src = [BATCH] if len(srcs) == 1 else [BATCH // 2, BATCH // 2]
    pools = []
    info = {}
    for src, n in zip(srcs, per_src):
        t0 = time.time()
        d, summ = load_source(a.dataset, src)
        crops = torch.from_numpy(d['crops']).to(dev)
        y = d['y']
        pos = np.nonzero(y == 1)[0]; neg = np.nonzero(y == 0)[0]
        pools.append((src, crops, pos, neg, n))
        info[src] = dict(candidates=int(len(y)), positives=int(len(pos)), negatives=int(len(neg)), per_batch=n,
                         expected_pos_per_batch=n / 11)
        log('loaded', src, info[src], f'{time.time() - t0:.0f}s')
    cfg = dict(dataset=a.dataset, arm=a.arm, seed=a.seed, steps=a.steps, batch=BATCH, lr=LR, wd=WD, size=SIZE,
               sources=info, model='torchvision resnet18 IMAGENET1K_V1, fc=Linear(512,1)', schedule='cosine, no warmup',
               aug='hflip .5, vflip .5, rot90 k~U{0..3}', norm=dict(mean=MEAN, std=STD), amp='bf16 autocast',
               torch=torch.__version__, torchvision=torchvision.__version__, host=os.uname().nodename,
               cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'), started=now())
    (rd / 'config.json').write_text(json.dumps(cfg, indent=1))

    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 1)
    model = model.to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / a.steps)))
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    model.train()
    hist = []
    acc = {'loss': 0.0, 'n': 0, 'npos': 0}
    acc_src = {p[0]: [0.0, 0] for p in pools}
    t0 = time.time()
    for step in range(a.steps):
        xs, ys, sid = [], [], []
        for j, (src, crops, pos, neg, n) in enumerate(pools):
            base = n // 11
            k = base + int(rng.random() < (n / 11 - base))
            ip = pos[rng.integers(0, len(pos), k)]
            ineg = neg[rng.integers(0, len(neg), n - k)]
            idx = torch.as_tensor(np.concatenate([ip, ineg]), device=dev)
            xs.append(crops[idx]); ys.append(torch.cat([torch.ones(k), torch.zeros(n - k)]))
            sid.append(torch.full((n,), j))
        x = torch.cat(xs).float().div_(255.0)
        yb = torch.cat(ys).to(dev)
        sidb = torch.cat(sid).to(dev)
        B = x.shape[0]
        hf = torch.rand(B, device=dev) < 0.5
        x = torch.where(hf[:, None, None, None], x.flip(3), x)
        vf = torch.rand(B, device=dev) < 0.5
        x = torch.where(vf[:, None, None, None], x.flip(2), x)
        kr = torch.randint(0, 4, (B,), device=dev)
        for r in (1, 2, 3):
            m = kr == r
            if m.any():
                x[m] = torch.rot90(x[m], r, dims=(2, 3))
        x = (x - mean) / std
        x = F.interpolate(x, size=(SIZE, SIZE), mode='bilinear', align_corners=False)
        x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logit = model(x).squeeze(1)
        lv = F.binary_cross_entropy_with_logits(logit.float(), yb, reduction='none')
        loss = lv.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        acc['loss'] += loss.item(); acc['n'] += 1; acc['npos'] += int(yb.sum().item())
        for j, p in enumerate(pools):
            m = sidb == j
            acc_src[p[0]][0] += lv[m].mean().item(); acc_src[p[0]][1] += 1
        if (step + 1) % 50 == 0 or step + 1 == a.steps:
            row = dict(step=step + 1, loss=acc['loss'] / acc['n'], lr=opt.param_groups[0]['lr'],
                       pos_frac=acc['npos'] / (acc['n'] * BATCH),
                       src_loss={s: v[0] / max(1, v[1]) for s, v in acc_src.items()}, sec=time.time() - t0)
            hist.append(row)
            if (step + 1) % 500 == 0 or step + 1 == a.steps:
                log(json.dumps(row))
            acc = {'loss': 0.0, 'n': 0, 'npos': 0}
            acc_src = {p[0]: [0.0, 0] for p in pools}
    (rd / 'loss.json').write_text(json.dumps(hist))
    torch.save(model.state_dict(), rd / 'model.pt')
    cfg['finished'] = now(); cfg['train_seconds'] = time.time() - t0
    (rd / 'config.json').write_text(json.dumps(cfg, indent=1))
    (rd / 'TRAIN_DONE').write_text(now() + '\n')
    log('done', rd, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
