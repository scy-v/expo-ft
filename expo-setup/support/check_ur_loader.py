"""CPU-only actual SFT checkpoint load; does not connect hardware or run inference."""
import os
os.environ['JAX_PLATFORMS']='cpu'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import faulthandler
import time
import yaml
from client.ur.policy import URPolicy
faulthandler.dump_traceback_later(60)
started=time.monotonic()
config=yaml.safe_load(Path('configs/ur/insert.yaml').read_text())
policy=URPolicy(config)
faulthandler.cancel_dump_traceback_later()
print(f'PASS CPU evaluation loader, actual checkpoint, {time.monotonic()-started:.1f}s; forward not run',flush=True)
