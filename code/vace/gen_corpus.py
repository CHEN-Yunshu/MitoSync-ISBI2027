#!/usr/bin/env python
"""Batch corpus generation: Wan2.1-VACE-1.3B + LoRA over a clip collection -> PNG frames + CTC image tifs.

Run with the VACE env:  $N/ext/vace_venv/bin/python $N/scripts/vace/gen_corpus.py \
    --dataset PhC-C2DL-PSC --lora $N/eval/vace_lora_PhC-C2DL-PSC_v1/step-2500.safetensors \
    --clips_root $N/data/vace_sim/PhC-C2DL-PSC/sim_v2_d10 --ctc_root $N/data/vace_sim/PhC-C2DL-PSC/sim_v2_d10_ctc \
    --gpu 0 [--steps 30 --cfg 5 --seed_base 0 --limit K --clips clip_000,clip_003 --readout_gate [--readout_thr 0.85]]

Per clip <clips_root>/<clip> (cond/frame_%03d.png, ref.png, labels/frame_%03d.png, meta.json):
  * pipeline + LoRA are loaded ONCE; 17 frames generated with seed = seed_base + index of the clip in the sorted
    clip list of clips_root (stable under --limit/--clips);
  * PNG frames -> <clips_root>/<clip>/gen/frame_%03d.png (+ gen/run_info.json);
  * CTC image tifs -> <ctc_root>/<clip>/t%03d.tif : gray = mean(R,G,B) of the generated 8-bit frame, mapped back
    with the dataset's inverse intensity mapping (intensity_mapping.json) to the native dtype/range;
  * resumable: a clip whose <ctc_root>/<clip>/t016.tif exists is skipped (unless --force); with --readout_gate a
    skipped clip that has no F1 in the manifest is only re-scored;
  * --readout_gate: Cellpose-SAM (readout_f1.py --serve, cellpose env subprocess, loaded once) segments gen/ frames,
    one-to-one IoU>=0.5 matching to labels/, mean requested-instance F1 < --readout_thr -> accepted=false and a
    <ctc_root>/<clip>/REJECTED marker file (tifs are still written so the run stays resumable);
  * <ctc_root>/manifest.json (per clip: seed, lora, steps, cfg, elapsed, readout F1, accepted) is rewritten
    atomically after every clip; <ctc_root>/accepted_dirs.txt lists the accepted <ctc_root>/<clip> dirs.
Per-clip exceptions are caught, logged and recorded (status=error); the run continues.
"""
import os, sys, json, time, argparse, glob, subprocess, traceback

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True, choices=["PhC-C2DL-PSC", "Fluo-N2DL-HeLa", "DIC-C2DH-HeLa", "Lead-BF"])
p.add_argument("--lora", required=True)
p.add_argument("--clips_root", required=True)
p.add_argument("--ctc_root", required=True)
p.add_argument("--gpu", default="0")
p.add_argument("--steps", type=int, default=30)
p.add_argument("--cfg", type=float, default=5.0)
p.add_argument("--seed_base", type=int, default=0)
p.add_argument("--limit", type=int, default=0, help="process at most this many (not yet done) clips; 0 = all")
p.add_argument("--clips", default=None, help="comma-separated explicit clip names (subset of clips_root)")
p.add_argument("--prompt", default=None, help="default: dataset prompt used for LoRA training")
p.add_argument("--negative_prompt", default="blurry, low quality, text, watermark, cartoon, painting, color, oversaturated, static, flicker")
p.add_argument("--height", type=int, default=None, help="default: from meta.json / cond frame size")
p.add_argument("--width", type=int, default=None)
p.add_argument("--num_frames", type=int, default=17)
p.add_argument("--lora_target", default="vace", choices=["vace", "dit"])
p.add_argument("--lora_alpha", type=float, default=1.0)
p.add_argument("--vace_scale", type=float, default=1.0)
p.add_argument("--sigma_shift", type=float, default=5.0)
p.add_argument("--weights", default=os.environ.get("VACE_WEIGHTS", os.path.join(os.environ.get("MITOSYNC_ROOT", os.environ.get("N", "./workspace")), "ext/vace/Wan2.1-VACE-1.3B")))
p.add_argument("--mapping", default=None, help="intensity_mapping.json (default $N/data/vace_clips/<dataset>/intensity_mapping.json)")
p.add_argument("--tif_compression", default="lzw", help="compression for t%%03d.tif (real CTC tifs are LZW)")
p.add_argument("--preview", action="store_true", help="also write gen/preview.mp4")
p.add_argument("--force", action="store_true", help="regenerate even if t016.tif exists")
p.add_argument("--readout_gate", action="store_true")
p.add_argument("--readout_thr", type=float, default=0.85)
p.add_argument("--readout_scale", type=float, default=1.0, help="upsample factor for Cellpose-SAM (small cells)")
p.add_argument("--readout_python", default=os.environ.get("CELLPOSE_PYTHON", sys.executable))
p.add_argument("--readout_gpu", default=None, help="GPU for the cellpose subprocess (default: same as --gpu)")
p.add_argument("--readout_only", action="store_true", help="skip generation entirely; only (re)score existing gen/ frames")
p.add_argument("--no_tiled", action="store_true")
args = p.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")

import numpy as np
import tifffile
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vace_common import list_pngs

N = os.environ.get("N", os.environ.get("MITOSYNC_ROOT", os.environ.get("N", "./workspace")))
PROMPTS = {"PhC-C2DL-PSC": "time-lapse microscopy video of cells, phase contrast",
           "Fluo-N2DL-HeLa": "time-lapse microscopy video of cells, fluorescence nuclei",
           "Lead-BF": "time-lapse microscopy video of cells, brightfield",
           "DIC-C2DH-HeLa": "time-lapse microscopy video of cells, differential interference contrast"}
HERE = os.path.dirname(os.path.abspath(__file__))


def log(*a):
    print(time.strftime("[%H:%M:%S]"), "[gen_corpus]", *a, flush=True)


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def load_mapping(path):
    m = json.load(open(path))
    dtype = np.dtype(m["source_dtype"])
    return dict(lo=float(m["lo"]), hi=float(m["hi"]), dtype=dtype, path=path)


def inverse_map(u8_gray, mp):
    """u8 (float, 0..255) -> native dtype via x = u8/255*(hi-lo)+lo (rounded, clipped to dtype range)."""
    x = u8_gray.astype(np.float64) / 255.0 * (mp["hi"] - mp["lo"]) + mp["lo"]
    info = np.iinfo(mp["dtype"])
    return np.clip(np.round(x), info.min, info.max).astype(mp["dtype"])


def load_pipe(weights, device="cuda"):
    import torch
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    W = weights
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=device,
        model_configs=[ModelConfig(path=os.path.join(W, "diffusion_pytorch_model.safetensors")),
                       ModelConfig(path=os.path.join(W, "models_t5_umt5-xxl-enc-bf16.pth")),
                       ModelConfig(path=os.path.join(W, "Wan2.1_VAE.pth"))],
        tokenizer_config=ModelConfig(path=os.path.join(W, "google", "umt5-xxl")))
    return pipe


class Readout:
    """Persistent Cellpose-SAM scorer (readout_f1.py --serve in the cellpose env)."""

    def __init__(self, python, gpu, scale):
        self.scale = scale
        cmd = [python, os.path.join(HERE, "readout_f1.py"), "--serve", "--gpu", str(gpu), "--scale", str(scale)]
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, text=True,
                                     bufsize=1, env=env)
        if not self._read().get("ready"):
            raise RuntimeError("readout server failed to start")
        log("readout server ready (pid", self.proc.pid, ")")

    def _read(self):
        """Next protocol line ('@@' + JSON); other stdout lines are ignored."""
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("readout server died")
            if line.startswith("@@"):
                return json.loads(line[2:])

    def score(self, frames_dir, labels_dir, n_frames):
        self.proc.stdin.write(json.dumps(dict(frames_dir=frames_dir, labels_dir=labels_dir, scale=self.scale, n_frames=n_frames)) + "\n")
        self.proc.stdin.flush()
        res = self._read()
        if "error" in res:
            raise RuntimeError("readout: " + res["error"])
        return res

    def close(self):
        try:
            self.proc.stdin.write(json.dumps(dict(quit=True)) + "\n"); self.proc.stdin.flush()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()


def main():
    t_all = time.time()
    os.makedirs(args.ctc_root, exist_ok=True)
    manifest_path = os.path.join(args.ctc_root, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    clips_entries = manifest.setdefault("clips", {})
    manifest["args"] = vars(args); manifest["dataset"] = args.dataset
    prompt = args.prompt or PROMPTS[args.dataset]
    mp = load_mapping(args.mapping or os.path.join(N, "data/vace_clips", args.dataset, "intensity_mapping.json"))
    log(f"intensity mapping {mp['path']}: lo={mp['lo']} hi={mp['hi']} dtype={mp['dtype']}")

    all_clips = sorted(d for d in os.listdir(args.clips_root) if d.startswith("clip_") and os.path.isdir(os.path.join(args.clips_root, d)))
    index = {c: i for i, c in enumerate(all_clips)}
    todo = all_clips if not args.clips else [c for c in args.clips.split(",") if c]
    missing = [c for c in todo if c not in index]
    if missing:
        log("WARNING unknown clips ignored:", missing); todo = [c for c in todo if c in index]

    def done(c):
        return os.path.exists(os.path.join(args.ctc_root, c, f"t{args.num_frames - 1:03d}.tif"))

    def needs_score(c):
        return args.readout_gate and clips_entries.get(c, {}).get("readout_f1") is None

    work = []
    for c in todo:
        if args.force or not done(c):
            work.append((c, "generate"))
        elif needs_score(c) and os.path.exists(os.path.join(args.clips_root, c, "gen", f"frame_{args.num_frames - 1:03d}.png")):
            work.append((c, "score"))
    if args.readout_only:
        work = [(c, "score") for c, _ in work]
    if args.limit:
        work = work[:args.limit]
    log(f"{len(all_clips)} clips in {args.clips_root}; {len(todo)} requested; {sum(1 for _, m in work if m == 'generate')} to generate, "
        f"{sum(1 for _, m in work if m == 'score')} to (re)score; seed_base={args.seed_base} steps={args.steps} cfg={args.cfg}")
    if not work:
        log("nothing to do"); return

    pipe = None
    if any(m == "generate" for _, m in work):
        import torch
        t0 = time.time()
        pipe = load_pipe(args.weights)
        target = pipe.vace if args.lora_target == "vace" else pipe.dit
        pipe.load_lora(target, args.lora, alpha=args.lora_alpha)
        log(f"pipeline + LoRA {args.lora} loaded in {time.time() - t0:.1f}s")
    readout = Readout(args.readout_python, args.readout_gpu if args.readout_gpu is not None else args.gpu, args.readout_scale) if args.readout_gate else None

    def write_accepted():
        acc = [os.path.join(args.ctc_root, c) for c, e in sorted(clips_entries.items()) if e.get("status") == "ok" and e.get("accepted", True)]
        tmp = os.path.join(args.ctc_root, "accepted_dirs.txt.tmp")
        with open(tmp, "w") as f:
            f.write("".join(d + "\n" for d in acc))
        os.replace(tmp, os.path.join(args.ctc_root, "accepted_dirs.txt"))
        return acc

    counts = dict(generated=0, scored=0, accepted=0, rejected=0, errors=0)
    gen_times = []
    for k, (clip, mode) in enumerate(work):
        t_clip = time.time()
        cdir = os.path.join(args.clips_root, clip); gdir = os.path.join(cdir, "gen"); odir = os.path.join(args.ctc_root, clip)
        entry = clips_entries.get(clip, {})
        entry.update(clip=clip, index=index[clip], seed=args.seed_base + index[clip], lora=args.lora, steps=args.steps, cfg=args.cfg,
                     prompt=prompt, status="running")
        if mode == "generate":  # regeneration (--force / missing tifs): drop stale scores + REJECTED marker from an earlier run [verifier fix]
            for stale in ("readout_f1", "readout_f1_min", "readout_f1_per_frame", "readout_n_gt_mean", "readout_n_pred_mean",
                          "readout_scale", "readout_thr", "readout_s", "accepted", "error"):
                entry.pop(stale, None)
            if os.path.exists(os.path.join(odir, "REJECTED")):
                os.remove(os.path.join(odir, "REJECTED"))
        try:
            if mode == "generate":
                meta = json.load(open(os.path.join(cdir, "meta.json"))) if os.path.exists(os.path.join(cdir, "meta.json")) else {}
                ctrl_files = list_pngs(os.path.join(cdir, "cond"))
                assert len(ctrl_files) >= args.num_frames, f"{len(ctrl_files)} cond frames < {args.num_frames}"
                control = [Image.open(f).convert("RGB") for f in ctrl_files[:args.num_frames]]
                W, H = control[0].size
                H = args.height or meta.get("height") or H; W = args.width or meta.get("width") or W
                assert H % 16 == 0 and W % 16 == 0, f"H/W must be multiples of 16, got {H}x{W}"
                for im in control:
                    assert im.size == (W, H), f"cond frame size {im.size} != ({W},{H})"
                ref = Image.open(os.path.join(cdir, "ref.png")).convert("RGB")
                assert ref.size == (W, H), f"ref size {ref.size} != ({W},{H})"
                torch.cuda.reset_peak_memory_stats()
                t1 = time.time()
                video = pipe(prompt=prompt, negative_prompt=args.negative_prompt, vace_video=control, vace_video_mask=None,
                             vace_reference_image=ref, vace_scale=args.vace_scale, seed=entry["seed"], height=H, width=W,
                             num_frames=args.num_frames, num_inference_steps=args.steps, cfg_scale=args.cfg,
                             sigma_shift=args.sigma_shift, tiled=not args.no_tiled)
                t_gen = time.time() - t1
                assert len(video) == args.num_frames, f"pipeline returned {len(video)} frames"
                os.makedirs(gdir, exist_ok=True); os.makedirs(odir, exist_ok=True)
                stats = []
                for i, fr in enumerate(video):
                    fr.save(os.path.join(gdir, f"frame_{i:03d}.png"))
                    g8 = np.asarray(fr.convert("RGB"), dtype=np.float32).mean(-1)
                    nat = inverse_map(g8, mp)
                    tifffile.imwrite(os.path.join(odir, f"t{i:03d}.tif"), nat, compression=args.tif_compression)
                    stats.append((float(nat.min()), float(nat.max()), float(nat.mean())))
                if args.preview:
                    from diffsynth.utils.data import save_video
                    save_video(video, os.path.join(gdir, "preview.mp4"), fps=8, quality=6)
                peak = torch.cuda.max_memory_allocated() / 2 ** 30
                info = dict(clip=clip, seed=entry["seed"], lora=args.lora, lora_target=args.lora_target, steps=args.steps, cfg=args.cfg,
                            prompt=prompt, negative_prompt=args.negative_prompt, height=H, width=W, num_frames=args.num_frames,
                            vace_scale=args.vace_scale, sigma_shift=args.sigma_shift, gen_s=round(t_gen, 1), peak_alloc_GB=round(peak, 2),
                            mapping=dict(lo=mp["lo"], hi=mp["hi"], dtype=str(mp["dtype"])),
                            tif_stats=dict(min=min(s[0] for s in stats), max=max(s[1] for s in stats), mean=float(np.mean([s[2] for s in stats]))))
                with open(os.path.join(gdir, "run_info.json"), "w") as f:
                    json.dump(info, f, indent=1)
                entry.update(gen_s=round(t_gen, 1), peak_alloc_GB=round(peak, 2), height=H, width=W, tif_stats=info["tif_stats"],
                             tif_dtype=str(mp["dtype"]), ctc_dir=odir, gen_dir=gdir)
                gen_times.append(t_gen); counts["generated"] += 1
            if readout is not None:
                r = readout.score(gdir, os.path.join(cdir, "labels"), args.num_frames)
                entry.update(readout_f1=round(r["f1_mean"], 4), readout_f1_min=round(r["f1_min"], 4), readout_f1_per_frame=r["f1_per_frame"],
                             readout_n_gt_mean=round(r["n_gt_mean"], 1), readout_n_pred_mean=round(r["n_pred_mean"], 1),
                             readout_scale=args.readout_scale, readout_thr=args.readout_thr, readout_s=r["seconds"])
                entry["accepted"] = bool(r["f1_mean"] >= args.readout_thr)
                counts["scored"] += 1
                marker = os.path.join(odir, "REJECTED")
                if entry["accepted"]:
                    counts["accepted"] += 1
                    if os.path.exists(marker):
                        os.remove(marker)
                else:
                    counts["rejected"] += 1
                    os.makedirs(odir, exist_ok=True)
                    with open(marker, "w") as f:
                        f.write(f"readout_f1={r['f1_mean']:.4f} < thr={args.readout_thr}\n")
            else:
                entry.setdefault("accepted", True)
            entry["status"] = "ok"
        except Exception as e:
            counts["errors"] += 1
            entry.update(status="error", error=f"{type(e).__name__}: {e}")
            log(f"ERROR {clip}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        entry["elapsed_s"] = round(time.time() - t_clip, 1)
        clips_entries[clip] = entry
        manifest["summary"] = dict(counts, n_entries=len(clips_entries), total_s=round(time.time() - t_all, 1))
        atomic_json(manifest_path, manifest)
        write_accepted()  # keep accepted_dirs.txt in sync with the manifest after every clip [verifier fix]
        log(f"[{k + 1}/{len(work)}] {clip} {mode} seed={entry['seed']} status={entry['status']} "
            f"gen_s={entry.get('gen_s', '-')} F1={entry.get('readout_f1', '-')} accepted={entry.get('accepted', '-')} elapsed={entry['elapsed_s']}s")

    if readout is not None:
        readout.close()
    acc = write_accepted()
    total = time.time() - t_all
    log(f"DONE {counts} total={total:.1f}s" + (f" mean_gen_s={np.mean(gen_times):.1f} per_clip_wall={total / max(len(work), 1):.1f}s" if gen_times else "")
        + f"; accepted dirs: {len(acc)} -> {os.path.join(args.ctc_root, 'accepted_dirs.txt')}")


if __name__ == "__main__":
    main()
