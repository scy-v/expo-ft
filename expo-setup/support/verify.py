"""Server dependency and GPU smoke checks; never connects to robot hardware."""
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
report = {"python": sys.version, "platform": platform.platform(), "checks": {}, "versions": {}}


def check(name, fn):
    try:
        value = fn()
        report["checks"][name] = {"status": "passed", "detail": str(value)}
        print(f"PASS {name}: {value}", flush=True)
    except Exception as exc:
        report["checks"][name] = {"status": "failed", "detail": repr(exc)}
        print(f"FAIL {name}: {exc!r}", flush=True)


for package in ["expo-ft", "openpi", "openpi-client", "jax", "jaxlib", "torch", "flax", "numpy", "lerobot", "transformers", "websockets"]:
    try:
        report["versions"][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        report["versions"][package] = "MISSING"

for module in ["openpi", "openpi_client", "lerobot", "flax", "optax", "h5py", "cv2", "tensorflow_probability.substrates.jax", "expo_ft.agents.alg", "expo_ft.data.replay_buffer", "expo_ft.env.env_client", "expo_ft.agents.vla.pi05"]:
    check(f"import:{module}", lambda module=module: getattr(importlib.import_module(module), "__file__", "imported"))


def config_check():
    from configs.model.expo_ft_pi_config import get_config
    from openpi.training.config import get_config as get_openpi_config
    cfg = get_config()
    assert cfg.model_cls == "EXPOLearner"
    return get_openpi_config(cfg.pi05_config_name).name


def jax_check():
    import jax
    import jax.numpy as jnp
    devices = jax.devices("gpu")
    assert devices
    for device in devices:
        with jax.default_device(device):
            x = jnp.ones((128, 128), dtype=jnp.float32)
            y = jax.jit(lambda a: a @ a)(x).block_until_ready()
            assert bool(jnp.all(y == 128))
    return [str(d) for d in devices]


def torch_check():
    import torch
    assert torch.cuda.is_available()
    for i in range(torch.cuda.device_count()):
        x = torch.ones((128, 128), device=f"cuda:{i}")
        assert torch.all(x @ x == 128).item()
    return {"cuda": torch.version.cuda, "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


check("config:EXPO-FT/OpenPI", config_check)
check("gpu:JAX-matmul", jax_check)
check("gpu:Torch-matmul", torch_check)
report["repo_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
report["openpi_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / "expo_ft/agents/vla/openpi", text=True).strip()
out = ROOT / "expo-setup/logs/versions-and-checks.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
raise SystemExit(any(v["status"] == "failed" for v in report["checks"].values()))
