# script/profile_model.py
import argparse, torch
import logging
import warnings
import time
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from torchvision.transforms.v2 import functional as F
from tqdm import tqdm
from main import LightningCLI
from training.lightning_module import LightningModule
from datasets.lightning_data_module import LightningDataModule

class ProfileCLI(LightningCLI):
    def __init__(self, args):
        super().__init__(
            LightningModule,
            LightningDataModule,
            subclass_mode_model=True,
            subclass_mode_data=True,
            save_config_callback=None,
            run=False,                     # 只构建，不训练
            args=args,
        )

def build_from_config(cfg_args, ckpt, device):
    cli = ProfileCLI(cfg_args)
    model = cli.model
    datamodule = cli.datamodule
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict):
            if "state_dict" in state:
                weights = state["state_dict"]
            elif "model" in state:
                weights = state["model"]
            else:
                weights = state
            loaded = model.load_state_dict(weights, strict=False)
            print("Missing keys:", loaded.missing_keys)
            print("Unexpected keys:", loaded.unexpected_keys)
        else:
            raise RuntimeError(f"Unsupported checkpoint format: {type(state)}")
    model.eval().to(device)
    return model, tuple(datamodule.img_size), datamodule

def _resize_img(img, img_size):
    if list(img.shape[-2:]) != list(img_size):
        img = F.resize(img, list(img_size), interpolation=F.InterpolationMode.BILINEAR)
    return img

@torch.inference_mode()
def measure_flops(model, datamodule, split, max_batches, img_size):
    device = next(model.parameters()).device
    stage = "fit" if split == "train" else "validate"
    datamodule.setup(stage)
    # Force batch_size=1 for variable-size validation images.
    if hasattr(datamodule, "dataloader_kwargs") and isinstance(
        datamodule.dataloader_kwargs, dict
    ):
        datamodule.dataloader_kwargs["batch_size"] = 1
    if split == "train":
        dataloader = datamodule.train_dataloader()
    elif split == "val":
        dataloader = datamodule.val_dataloader()
    else:
        raise ValueError("split must be 'train' or 'val'")

    total_flops = 0.0
    total_imgs = 0

    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
    for batch_idx, batch in enumerate(
        tqdm(
            dataloader,
            total=total_batches,
            desc=f"FLOPs ({split})",
        )
    ):
        if max_batches is not None and batch_idx >= max_batches:
            break

        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        if isinstance(imgs, (list, tuple)):
            if len(imgs) != 1:
                raise RuntimeError(
                    "FLOPs averaging expects batch_size=1 for variable-size images."
                )
            img = imgs[0]
        else:
            img = imgs

        if not torch.is_tensor(img):
            raise TypeError(f"Unsupported batch img type: {type(img)}")

        img = _resize_img(img, img_size)
        img = img.unsqueeze(0).to(device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logging.getLogger("fvcore").setLevel(logging.ERROR)
            flops = FlopCountAnalysis(model, img)
        total_flops += flops.total()
        total_imgs += 1

    if total_imgs == 0:
        raise RuntimeError("No images found for FLOPs averaging.")

    avg_flops = total_flops / total_imgs
    print(f"Avg FLOPs: {avg_flops / 1e9:.2f} GFLOPs/image over {total_imgs} images")
    print(parameter_count_table(model))

@torch.inference_mode()
def measure_throughput(model, datamodule, split, max_batches, img_size, repeat, warmup):
    device = next(model.parameters()).device
    stage = "fit" if split == "train" else "validate"
    datamodule.setup(stage)
    if hasattr(datamodule, "dataloader_kwargs") and isinstance(
        datamodule.dataloader_kwargs, dict
    ):
        datamodule.dataloader_kwargs["batch_size"] = 1
    if split == "train":
        dataloader = datamodule.train_dataloader()
    elif split == "val":
        dataloader = datamodule.val_dataloader()
    else:
        raise ValueError("split must be 'train' or 'val'")

    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
    timings = []
    warmup_iters = warmup

    for batch_idx, batch in enumerate(
        tqdm(
            dataloader,
            total=total_batches,
            desc=f"FPS ({split})",
        )
    ):
        if max_batches is not None and batch_idx >= max_batches:
            break

        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        if isinstance(imgs, (list, tuple)):
            if len(imgs) != 1:
                raise RuntimeError(
                    "Throughput expects batch_size=1 for variable-size images."
                )
            img = imgs[0]
        else:
            img = imgs

        if not torch.is_tensor(img):
            raise TypeError(f"Unsupported batch img type: {type(img)}")

        img = _resize_img(img, img_size)
        img = img.unsqueeze(0).to(device)
        img = img.repeat(repeat, 1, 1, 1)

        if batch_idx < warmup_iters:
            _ = model(img)
            continue

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        _ = model(img)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.time() - start)

    if not timings:
        raise RuntimeError("No timing samples collected for throughput.")

    timing = torch.as_tensor(timings, dtype=torch.float32)
    fps = (repeat / timing.mean()).item()
    print(f"Throughput: {fps:.2f} images/s (repeat={repeat})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--flops-split", choices=["train", "val"], default="val")
    parser.add_argument("--flops-max-batches", type=int, default=None)
    parser.add_argument("--fps-split", choices=["train", "val"], default="val")
    parser.add_argument("--fps-max-batches", type=int, default=None)
    args, unknown = parser.parse_known_args()

    cli_args = ["-c", args.config] + unknown + ["--compile_disabled"]
    model, img_size, datamodule = build_from_config(cli_args, args.ckpt, args.device)
    measure_flops(
        model,
        datamodule,
        args.flops_split,
        args.flops_max_batches,
        img_size,
    )
    measure_throughput(
        model,
        datamodule,
        args.fps_split,
        args.fps_max_batches,
        img_size,
        repeat=args.batch_size,
        warmup=args.warmup,
    )
