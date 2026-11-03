#!/usr/bin/env python
"""LoRA training for Wan2.1-VACE-1.3B on LOCAL weights with a lossless PNG-folder dataset.

Thin wrapper around the official DiffSynth-Studio examples/wanvideo/model_training/train.py
(kept verbatim next to this file as train.py). It reuses WanTrainingModule / wan_parser / launch_training_task
and only replaces the dataset so that a metadata.csv cell may point to a FOLDER of PNG frames
(the official UnifiedDataset only accepts mp4/gif/single-image files, which are lossy or palettised).

metadata.csv columns (paths relative to --dataset_base_path):
  video                 folder of PNG frames (target clip)  | or an mp4 (official loader, lossy)
  vace_video            folder of PNG frames (control clip) | or an mp4
  vace_reference_image  single PNG
  prompt                text
All folder keys of one row are sampled with the SAME frame window [start, start+num_frames) (start=0 unless
--random_clip_start). Frames are sorted by filename. Fewer than num_frames frames -> truncated to 4k+1.

Launch (see train_lora.sh):
  accelerate launch --num_processes 1 train_lora.py --dataset_base_path D --dataset_metadata_path D/metadata.csv \
     --weights_dir $N/ext/vace/Wan2.1-VACE-1.3B --height 576 --width 720 --num_frames 17 --lora_rank 32 \
     --learning_rate 1e-4 --num_epochs 1 --dataset_repeat 8 --output_path OUT --lora_base_model vace \
     --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --extra_inputs "vace_video,vace_reference_image" \
     --remove_prefix_in_ckpt "pipe.vace." --use_gradient_checkpointing --enable_csv_log
"""
import os, sys, json, glob, random
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch, accelerate
from PIL import Image
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import ImageCropAndResize
from diffsynth.diffusion import ModelLogger, launch_training_task, launch_data_process_task
import train as official  # official train.py (WanTrainingModule, wan_parser)


class FolderOrFileVideoDataset(UnifiedDataset):
    """UnifiedDataset whose data_file_keys may be PNG folders (lossless) or files (official loaders)."""

    def __init__(self, *a, num_frames=81, height=None, width=None, max_pixels=1920 * 1080,
                 random_clip_start=False, frame_stride=1, **kw):
        super().__init__(*a, **kw)
        self.num_frames = num_frames
        self.random_clip_start = random_clip_start
        self.frame_stride = frame_stride
        self.frame_processor = ImageCropAndResize(height, width, max_pixels, 16, 16)

    def _abs(self, path):
        return path if os.path.isabs(path) else os.path.join(self.base_path, path)

    def _folder_frames(self, folder):
        files = sorted(glob.glob(os.path.join(folder, "*.png")))
        if not files:
            raise FileNotFoundError(f"no PNG frames in {folder}")
        return files

    def __getitem__(self, data_id):
        if self.load_from_cache:
            return super().__getitem__(data_id)
        data = self.data[data_id % len(self.data)].copy()
        folder_keys = [k for k in self.data_file_keys if k in data and os.path.isdir(self._abs(data[k]))]
        # one shared window for all folder keys of this row
        start = 0
        n = self.num_frames
        if folder_keys:
            lens = [len(self._folder_frames(self._abs(data[k]))) for k in folder_keys]
            avail = (min(lens) - 1) // self.frame_stride + 1
            if avail < n:
                n = avail
                while n > 1 and n % 4 != 1:
                    n -= 1
            if self.random_clip_start:
                max_start = min(lens) - (n - 1) * self.frame_stride - 1
                start = random.randint(0, max(max_start, 0))
        for key in self.data_file_keys:
            if key not in data:
                continue
            if key in folder_keys:
                files = self._folder_frames(self._abs(data[key]))
                idx = [start + i * self.frame_stride for i in range(n)]
                data[key] = [self.frame_processor(Image.open(files[i]).convert("RGB")) for i in idx]
            elif key in self.special_operator_map:
                data[key] = self.special_operator_map[key](data[key])
            else:
                data[key] = self.main_data_operator(data[key])
        return data


def build_parser():
    parser = official.wan_parser()
    parser.add_argument("--weights_dir", type=str, default=None,
                        help="Local Wan2.1-VACE-1.3B folder; sets --model_paths and --tokenizer_path if they are not given.")
    parser.add_argument("--random_clip_start", action="store_true", help="random temporal window per sample (folders only)")
    parser.add_argument("--frame_stride", type=int, default=1, help="temporal stride inside a PNG folder")
    parser.add_argument("--max_steps", type=int, default=None, help="stop after this many optimizer steps (debug)")
    return parser


def main():
    args = build_parser().parse_args()
    if args.weights_dir is not None:
        W = args.weights_dir
        if args.model_paths is None:
            args.model_paths = json.dumps([
                os.path.join(W, "diffusion_pytorch_model.safetensors"),
                os.path.join(W, "models_t5_umt5-xxl-enc-bf16.pth"),
                os.path.join(W, "Wan2.1_VAE.pth"),
            ])
        if args.tokenizer_path is None:
            args.tokenizer_path = os.path.join(W, "google", "umt5-xxl")
    assert args.model_paths is not None or args.model_id_with_origin_paths is not None

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = FolderOrFileVideoDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path, max_pixels=args.max_pixels, height=args.height, width=args.width,
            height_division_factor=16, width_division_factor=16,
            num_frames=args.num_frames, time_division_factor=4, time_division_remainder=1,
        ),
        num_frames=args.num_frames, height=args.height, width=args.width, max_pixels=args.max_pixels,
        random_clip_start=args.random_clip_start, frame_stride=args.frame_stride,
    )
    if args.max_steps is not None:
        dataset.max_data_items = args.max_steps * args.gradient_accumulation_steps
    model = official.WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        quant_options=args.quant_options,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train_lora] trainable params: {n_train/1e6:.2f} M; dataset items/epoch: {len(dataset)}")
    model_logger = ModelLogger(
        args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log, enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project, enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project, enable_csv_log=args.enable_csv_log,
    )
    launcher = launch_data_process_task if args.task.endswith(":data_process") else launch_training_task
    launcher(accelerator, dataset, model, model_logger, args=args)
    if accelerator.is_main_process:
        print(f"[train_lora] peak VRAM alloc {torch.cuda.max_memory_allocated()/2**30:.2f} GB, reserved {torch.cuda.max_memory_reserved()/2**30:.2f} GB")


if __name__ == "__main__":
    main()
