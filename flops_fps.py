# script/profile_model.py
import argparse, torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
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
    return model, tuple(cli.datamodule.img_size)

@torch.inference_mode()
def measure_flops(model, img_size):
    dummy = torch.zeros(1, 3, *img_size, device=next(model.parameters()).device)
    flops = FlopCountAnalysis(model, dummy)
    print(f"FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
    print(parameter_count_table(model))

@torch.inference_mode()
def measure_throughput(model, img_size, batch, warmup, iters):
    device = next(model.parameters()).device
    x = torch.randn(batch, 3, *img_size, device=device) * 255
    for _ in range(warmup):
        _ = model(x)
    torch.cuda.synchronize() if device.type == "cuda" else None

    times = []
    for _ in range(iters):
        if device.type == "cuda":
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record(); _ = model(x); end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # ms
        else:
            t0 = torch.cuda.Event(enable_timing=False); t0.record()
            _ = model(x); t1 = torch.cuda.Event(enable_timing=False); t1.record()
            times.append((t1.cpu_time_total - t0.cpu_time_total) / 1e6)

    avg_ms = sum(times) / len(times)
    print(f"Throughput: {batch * 1000 / avg_ms:.2f} images/s, avg latency: {avg_ms:.2f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args, unknown = parser.parse_known_args()

    # 把剩余 CLI 参数透传给 Lightning 配置（如 --data.path、--model.*、--compile_disabled True 等）
    cli_args = ["-c", args.config] + unknown + ["--compile_disabled"]
    model, img_size = build_from_config(cli_args, args.ckpt, args.device)
    measure_flops(model, img_size)
    measure_throughput(model, img_size, args.batch_size, args.warmup, args.iters)
