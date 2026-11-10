import os, json, platform
out = {"env_tpu": {k: v for k, v in os.environ.items() if "TPU" in k or "XLA" in k or "ACCEL" in k.upper()}}
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda"] = bool(torch.cuda.is_available())
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        out["torch_xla"] = torch_xla.__version__ if hasattr(torch_xla, "__version__") else "?"
        d = xm.xla_device()
        a = torch.randn(64, 64).to(d)
        b = (a @ a.T).sum()
        xm.wait_device_ops()
        out["xla_matmul_ok"] = float(b.cpu())
    except Exception as e:
        out["xla_err"] = repr(e)[:300]
except Exception as e:
    out["torch_err"] = repr(e)[:200]
print("TPUPROBE " + json.dumps(out), flush=True)
