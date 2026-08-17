#!/usr/bin/env python3
"""
Sequential ImageNet depth sweep: plain nets (no skip) at depths 8, 14, 20, 26,
34, 56, 74, then ResNet-74 (with skips). Built for multi-day runs: atomic
checkpoints, automatic resume, append-only JSONL metrics, and NPZ export.

Layout (under --out, default ./runs):
    runs/
      val_subset_indices.npy          fixed val subset shared by all models
      run.log                         full text log
      plain8/
        metrics.jsonl                 one JSON record per eval step (every 100)
        metrics.npz                   arrays exported at end of run (and on resume)
        ckpt_step001000.pt            model weights only (fp32), every 1000 steps
        resume_latest.pt              full training state (rolling, atomic)
        final.pt                      weights at final step
        DONE                          marker: this model finished
      plain14/ ...
      resnet74/ ...

Resume: just rerun the same command. Finished models are skipped (DONE marker),
the in-progress model resumes from resume_latest.pt, and metrics.jsonl is
truncated back to the resume step so there are no duplicate/orphan records.
Note: dataloader shuffle order after a resume is a fresh shuffle, not a replay
of the interrupted epoch -- fine for this purpose.

Typical launch (survives ssh disconnects):
    tmux new -s sweep
    python train_depth_sweep.py --data /home/stephen/imagenet --out runs
"""

import argparse
import json
import math
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.models.resnet import ResNet, BasicBlock

# ----------------------------------------------------------------------------
# Schedule: (name, depth, use_skip, total_iterations)
# 40k for the shallowest, +10k for each subsequent run.
# ----------------------------------------------------------------------------
SCHEDULE = [
    ("plain8",    8, False,  40_000),
    ("plain14",  14, False,  50_000),
    ("plain20",  20, False,  60_000),
    ("plain26",  26, False,  70_000),
    ("plain34",  34, False,  80_000),
    ("plain56",  56, False,  90_000),
    ("plain74",  74, False, 100_000),
    ("resnet74", 74, True,  110_000),
]

LAYER_CFG = {          # torchvision ResNet block counts per stage
    8:  [1, 1, 1, 1],  # + layer4 -> Identity surgery (3 blocks, depth 8)
    14: [2, 1, 1, 2],
    20: [2, 2, 3, 2],
    26: [3, 3, 3, 3],
    34: [4, 4, 4, 4],
    56: [3, 4, 17, 3],
    74: [3, 4, 26, 3],
}


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
class PlainBasicBlock(BasicBlock):
    """torchvision BasicBlock with the skip connection removed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.downsample = None  # drop the 1x1 projection params too

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))  # no identity add
        return out


def make_net(depth_target, use_skip, num_classes=1000):
    block = BasicBlock if use_skip else PlainBasicBlock
    model = ResNet(block, LAYER_CFG[depth_target], num_classes=num_classes)
    if depth_target == 8:
        # [1,1,1,1] is depth 10; drop stage 4 to reach depth 8.
        model.layer4 = nn.Identity()
        model.fc = nn.Linear(256, num_classes)
    return model


def depth(m):
    return sum(1 for x in m.modules()
               if isinstance(x, nn.Linear)
               or (isinstance(x, nn.Conv2d) and x.kernel_size != (1, 1)))


class BlockMonitor:
    """Activation stats (forward hooks) + per-block grad norms."""

    def __init__(self, model):
        stages = [model.layer1, model.layer2, model.layer3, model.layer4]
        self.blocks = [b for s in stages if isinstance(s, nn.Sequential) for b in s]
        self.act = {}
        self.handles = [b.register_forward_hook(self._hook(i))
                        for i, b in enumerate(self.blocks)]

    def _hook(self, i):
        def fn(mod, inp, out):
            with torch.no_grad():
                self.act[i] = (out.mean().item(), out.std().item())
        return fn

    def grad_norms(self):
        return [math.sqrt(sum(float(p.grad.norm()) ** 2
                              for p in b.parameters() if p.grad is not None))
                for b in self.blocks]

    def act_stats(self):
        return [list(self.act.get(i, (float("nan"), float("nan"))))
                for i in range(len(self.blocks))]


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------
def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)  # atomic on POSIX


def log(msg, logfile=None):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if logfile is not None:
        with open(logfile, "a") as f:
            f.write(line + "\n")


def weight_norm(model):
    with torch.no_grad():
        return math.sqrt(sum(float(p.norm()) ** 2 for p in model.parameters()))


def truncate_jsonl(path: Path, max_step: int):
    """Drop metric records past the resume step so the log stays consistent."""
    if not path.exists():
        return
    kept = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                if json.loads(line)["step"] <= max_step:
                    kept.append(line)
            except (json.JSONDecodeError, KeyError):
                continue  # drop torn final line from a hard kill
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(kept) + ("\n" if kept else ""))
    os.replace(tmp, path)


def export_npz(jsonl_path: Path, npz_path: Path):
    recs = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not recs:
        return
    scalar_keys = ["step", "wall_time", "train_loss", "train_loss_avg",
                   "train_top1", "lr", "grad_norm", "weight_norm",
                   "imgs_per_sec", "val_top1", "val_top5", "val_err1",
                   "val_loss"]
    out = {k: np.array([r.get(k, np.nan) for r in recs], dtype=np.float64)
           for k in scalar_keys}
    out["block_grad_norms"] = np.array([r["block_grad_norms"] for r in recs])
    out["block_act_mean"] = np.array([[a[0] for a in r["block_acts"]] for r in recs])
    out["block_act_std"] = np.array([[a[1] for a in r["block_acts"]] for r in recs])
    # full-val records are sparser; store step-aligned with NaN elsewhere
    for k in ["full_val_top1", "full_val_top5", "full_val_err1", "full_val_loss"]:
        out[k] = np.array([r.get(k, np.nan) for r in recs], dtype=np.float64)
    np.savez(npz_path, **out)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def build_datasets(data_root: Path):
    import torchvision.datasets as dsets
    import torchvision.transforms as T
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_ds = dsets.ImageFolder(
        data_root / "ILSVRC/Data/CLS-LOC/train",
        T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
                   T.ToTensor(), norm]))
    val_ds = dsets.ImageFolder(
        data_root / "ILSVRC/Data/CLS-LOC/val",
        T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm]))
    return train_ds, val_ds


def get_val_subset(val_ds, out_dir: Path, n=5000, seed=0):
    """Fixed, class-stratified-ish (random but seeded) subset shared by all
    models, persisted to disk so restarts and later analysis agree."""
    idx_path = out_dir / "val_subset_indices.npy"
    if idx_path.exists():
        idx = np.load(idx_path)
    else:
        g = np.random.default_rng(seed)
        idx = np.sort(g.choice(len(val_ds), size=min(n, len(val_ds)),
                               replace=False))
        np.save(idx_path, idx)
    return Subset(val_ds, idx.tolist())


def infinite_batches(loader):
    while True:
        yield from loader


# ----------------------------------------------------------------------------
# Eval
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, amp):
    """Returns (top1, top5, mean_loss). Restores train mode afterward."""
    was_training = model.training
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    top1 = top5 = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device, enabled=amp):
            logits = model(x)
            loss_sum += crit(logits.float(), y).item()
        _, pred5 = logits.topk(5, dim=1)
        correct = pred5.eq(y.unsqueeze(1))
        top1 += correct[:, 0].sum().item()
        top5 += correct.any(dim=1).sum().item()
        total += y.numel()
    if was_training:
        model.train()
    return top1 / total, top5 / total, loss_sum / total


# ----------------------------------------------------------------------------
# Training one model
# ----------------------------------------------------------------------------
def train_one(name, depth_target, use_skip, total_steps, args, train_loader,
              quick_val_loader, full_val_loader, device, logfile):
    out_dir = args.out / name
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / "DONE"
    if done_marker.exists():
        log(f"{name}: DONE marker found, skipping.", logfile)
        return

    metrics_path = out_dir / "metrics.jsonl"
    resume_path = out_dir / "resume_latest.pt"
    amp = (device == "cuda") and not args.no_amp

    torch.manual_seed(args.seed)
    model = make_net(depth_target, use_skip).to(device)
    d = depth(model)
    n_params = sum(p.numel() for p in model.parameters())
    assert d == depth_target, f"{name}: built depth {d}, expected {depth_target}"

    mon = BlockMonitor(model)
    opt = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9,
                          weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / args.warmup)
        * 0.5 * (1 + math.cos(math.pi * s / total_steps)))
    scaler = torch.amp.GradScaler(device, enabled=amp)
    crit = nn.CrossEntropyLoss()

    start_step = 0
    if resume_path.exists():
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        torch.set_rng_state(ck["cpu_rng"])
        if device == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        np.random.set_state(ck["np_rng"])
        start_step = ck["step"]
        truncate_jsonl(metrics_path, start_step)
        log(f"{name}: resumed at step {start_step}/{total_steps}", logfile)
    else:
        log(f"{name}: fresh start. depth={d} params={n_params/1e6:.2f}M "
            f"steps={total_steps}", logfile)

    def save_resume(step):
        atomic_save({
            "name": name, "step": step,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": (torch.cuda.get_rng_state_all()
                         if device == "cuda" else None),
            "np_rng": np.random.get_state(),
            "torch_version": torch.__version__,
        }, resume_path)

    model.train()
    batches = infinite_batches(train_loader)
    loss_buf = deque(maxlen=args.eval_every)  # detached GPU scalars
    step = start_step
    t_last, imgs_since = time.time(), 0

    try:
        while step < total_steps:
            x, y = next(batches)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=amp):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_buf.append(loss.detach())
            step += 1
            imgs_since += y.numel()

            # ---- lightweight eval + metrics record ----
            if step % args.eval_every == 0 or step == total_steps:
                now = time.time()
                v1, v5, vloss = evaluate(model, quick_val_loader, device, amp)
                with torch.no_grad():
                    tr_top1 = (logits.argmax(1) == y).float().mean().item()
                    tr_loss_avg = torch.stack(tuple(loss_buf)).mean().item()
                rec = {
                    "step": step,
                    "wall_time": now,
                    "train_loss": loss.item(),
                    "train_loss_avg": tr_loss_avg,
                    "train_top1": tr_top1,
                    "lr": sched.get_last_lr()[0],
                    "grad_norm": float(gnorm),
                    "weight_norm": weight_norm(model),
                    "imgs_per_sec": imgs_since / max(now - t_last, 1e-9),
                    "block_grad_norms": mon.grad_norms(),
                    "block_acts": mon.act_stats(),
                    "val_top1": v1, "val_top5": v5,
                    "val_err1": 1.0 - v1, "val_loss": vloss,
                }
                t_last, imgs_since = time.time(), 0

                # ---- checkpoint + full validation ----
                if step % args.ckpt_every == 0 or step == total_steps:
                    f1, f5, floss = evaluate(model, full_val_loader, device, amp)
                    rec.update({"full_val_top1": f1, "full_val_top5": f5,
                                "full_val_err1": 1.0 - f1,
                                "full_val_loss": floss})
                    cpu_sd = {k: v.detach().cpu()
                              for k, v in model.state_dict().items()}
                    atomic_save(cpu_sd, out_dir / f"ckpt_step{step:06d}.pt")
                    save_resume(step)
                    log(f"{name} s{step}/{total_steps} "
                        f"loss={tr_loss_avg:.3f} quick_val@1={v1:.4f} "
                        f"FULL val@1={f1:.4f} val@5={f5:.4f} "
                        f"lr={rec['lr']:.4f}", logfile)
                else:
                    log(f"{name} s{step}/{total_steps} "
                        f"loss={tr_loss_avg:.3f} val@1={v1:.4f} "
                        f"gnorm={float(gnorm):.2f}", logfile)

                with open(metrics_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")

    except KeyboardInterrupt:
        log(f"{name}: interrupted at step {step}, saving resume state...",
            logfile)
        save_resume(step)
        export_npz(metrics_path, out_dir / "metrics.npz")
        raise

    # ---- finished ----
    atomic_save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                out_dir / "final.pt")
    export_npz(metrics_path, out_dir / "metrics.npz")
    done_marker.write_text(f"finished step {step} at "
                           f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    for h in mon.handles:
        h.remove()
    del model, opt, sched, scaler, mon
    if device == "cuda":
        torch.cuda.empty_cache()
    log(f"{name}: DONE.", logfile)


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("/home/stephen/imagenet"))
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=100,
                   help="quick val (fixed subset) + metrics record cadence")
    p.add_argument("--ckpt-every", type=int, default=1000,
                   help="weight checkpoint + full val cadence")
    p.add_argument("--val-subset", type=int, default=5000,
                   help="size of the fixed quick-val subset")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-amp", action="store_true",
                   help="train in pure fp32 (slower)")
    p.add_argument("--models", nargs="*", default=None,
                   help="subset of model names to run, e.g. --models plain8 resnet74")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    logfile = args.out / "run.log"
    log(f"torch {torch.__version__} on {device}; out={args.out}", logfile)

    # make SIGTERM (e.g. system shutdown) behave like Ctrl-C -> clean resume save
    signal.signal(signal.SIGTERM, lambda *a: (_ for _ in ()).throw(KeyboardInterrupt))

    train_ds, val_ds = build_datasets(args.data)
    quick_val_ds = get_val_subset(val_ds, args.out, n=args.val_subset,
                                  seed=args.seed)
    train_loader = DataLoader(train_ds, args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True, persistent_workers=True)
    quick_val_loader = DataLoader(quick_val_ds, args.batch, shuffle=False,
                                  num_workers=max(2, args.workers // 2),
                                  pin_memory=True)
    full_val_loader = DataLoader(val_ds, args.batch, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)
    log(f"train={len(train_ds)} val={len(val_ds)} "
        f"quick_val={len(quick_val_ds)}", logfile)

    schedule = SCHEDULE
    if args.models:
        schedule = [s for s in SCHEDULE if s[0] in args.models]
        assert schedule, f"no schedule entries match {args.models}"

    t0 = time.time()
    for name, d, skip, steps in schedule:
        train_one(name, d, skip, steps, args, train_loader, quick_val_loader,
                  full_val_loader, device, logfile)
    log(f"ALL DONE in {(time.time()-t0)/3600:.1f} h", logfile)


if __name__ == "__main__":
    main()