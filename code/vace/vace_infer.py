#!/usr/bin/env python
"""Wan2.1-VACE-1.3B inference from LOCAL weights (DiffSynth-Studio 2.1.x API).

Example:
  $N/ext/vace_venv/bin/python $N/scripts/vace/vace_infer.py \
      --control_dir ctrl_png_dir --ref ref.png --prompt "..." --out_dir out \
      --height 576 --width 720 --num_frames 17 --steps 30 --cfg 5 --seed 0 --gpu 0 [--lora ckpt.safetensors]

Outputs: <out_dir>/frame_%03d.png (lossless), <out_dir>/preview.mp4, <out_dir>/run_info.json
"""
import os, sys, argparse, json, time

p = argparse.ArgumentParser()
p.add_argument("--control_dir", required=True, help="folder of RGB PNG control frames (sorted by name)")
p.add_argument("--ref", default=None, help="reference PNG (RGB). Optional but recommended.")
p.add_argument("--mask_dir", default=None, help="optional folder of PNG masks (white=regenerate, black=keep control pixels)")
p.add_argument("--prompt", default="phase-contrast microscopy video of living cells, time-lapse, high detail")
p.add_argument("--negative_prompt", default="blurry, low quality, text, watermark, cartoon, painting, color, oversaturated, static, flicker")
p.add_argument("--out_dir", required=True)
p.add_argument("--height", type=int, default=576)
p.add_argument("--width", type=int, default=720)
p.add_argument("--num_frames", type=int, default=None, help="4k+1; default = number of control frames")
p.add_argument("--steps", type=int, default=30)
p.add_argument("--cfg", type=float, default=5.0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--lora", default=None, help="LoRA .safetensors from train_lora (keys relative to pipe.vace or pipe.dit)")
p.add_argument("--lora_target", default="vace", choices=["vace", "dit"], help="which sub-model the LoRA was trained on")
p.add_argument("--lora_alpha", type=float, default=1.0)
p.add_argument("--vace_scale", type=float, default=1.0)
p.add_argument("--sigma_shift", type=float, default=5.0)
p.add_argument("--gpu", default="0")
p.add_argument("--weights", default=os.environ.get("VACE_WEIGHTS", "STUDY_ROOT/ext/vace/Wan2.1-VACE-1.3B"))
p.add_argument("--no_tiled", action="store_true", help="disable VAE tiling")
p.add_argument("--fps", type=int, default=8)
args = p.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")

import torch
import numpy as np
from PIL import Image
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vace_common import list_pngs


def load_pipe(weights, device="cuda"):
    W = weights
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=os.path.join(W, "diffusion_pytorch_model.safetensors")),  # -> dit + vace
            ModelConfig(path=os.path.join(W, "models_t5_umt5-xxl-enc-bf16.pth")),       # -> text_encoder
            ModelConfig(path=os.path.join(W, "Wan2.1_VAE.pth")),                        # -> vae
        ],
        tokenizer_config=ModelConfig(path=os.path.join(W, "google", "umt5-xxl")),
    )
    return pipe


def main():
    t_all = time.time()
    ctrl_files = list_pngs(args.control_dir)
    num_frames = args.num_frames or len(ctrl_files)
    assert (num_frames - 1) % 4 == 0, f"num_frames must be 4k+1, got {num_frames}"
    assert args.height % 16 == 0 and args.width % 16 == 0, "height/width must be multiples of 16"
    assert len(ctrl_files) >= num_frames, f"only {len(ctrl_files)} control frames, need {num_frames}"
    control = [Image.open(f).convert("RGB") for f in ctrl_files[:num_frames]]
    for im in control:
        assert im.size == (args.width, args.height), f"control frame size {im.size} != ({args.width},{args.height}); no implicit resize to keep aspect"
    ref = None
    if args.ref:
        ref = Image.open(args.ref).convert("RGB")
        assert ref.size == (args.width, args.height), f"ref size {ref.size} != ({args.width},{args.height})"
    mask = None
    if args.mask_dir:
        mask_files = list_pngs(args.mask_dir)[:num_frames]
        mask = [Image.open(f).convert("RGB") for f in mask_files]

    t0 = time.time()
    pipe = load_pipe(args.weights)
    t_load = time.time() - t0
    if args.lora:
        target = pipe.vace if args.lora_target == "vace" else pipe.dit
        pipe.load_lora(target, args.lora, alpha=args.lora_alpha)
        print(f"[vace_infer] LoRA {args.lora} fused into pipe.{args.lora_target}")

    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        vace_video=control,
        vace_video_mask=mask,
        vace_reference_image=ref,
        vace_scale=args.vace_scale,
        seed=args.seed,
        height=args.height, width=args.width, num_frames=num_frames,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg,
        sigma_shift=args.sigma_shift,
        tiled=not args.no_tiled,
    )
    t_gen = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 2**30
    peak_res = torch.cuda.max_memory_reserved() / 2**30

    os.makedirs(args.out_dir, exist_ok=True)
    for i, fr in enumerate(video):
        fr.save(os.path.join(args.out_dir, f"frame_{i:03d}.png"))
    save_video(video, os.path.join(args.out_dir, "preview.mp4"), fps=args.fps, quality=6)
    info = dict(vars(args), num_frames=num_frames, n_out_frames=len(video),
                load_s=round(t_load, 1), gen_s=round(t_gen, 1), total_s=round(time.time() - t_all, 1),
                peak_alloc_GB=round(peak, 2), peak_reserved_GB=round(peak_res, 2),
                out_size=list(video[0].size))
    with open(os.path.join(args.out_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("[vace_infer] " + json.dumps({k: info[k] for k in ["n_out_frames", "load_s", "gen_s", "total_s", "peak_alloc_GB", "peak_reserved_GB", "out_size"]}))


if __name__ == "__main__":
    main()
