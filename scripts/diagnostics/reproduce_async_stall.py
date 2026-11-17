"""Real batch64/UTD20 learner with cached observations; no robot or formal outputs."""
import os, sys, re, shlex, textwrap, time, threading, traceback, json
import argparse
from pathlib import Path
root=Path(__file__).resolve().parents[2]; os.chdir(root); sys.path.insert(0,str(root))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--updates', type=int, default=3)
parser.add_argument('--wait-mode', choices=['serial', 'batched'], default='serial')
parser.add_argument('--profile', action='store_true', help='Use CUDA profiler capture-range APIs for the first update')
options = parser.parse_args()
if options.updates < 1:
 parser.error('--updates must be positive')
output = options.output.resolve()
output.mkdir(parents=True, exist_ok=False)
block=next(b for b in re.findall(r'```bash\n(.*?)```',Path('expo-setup/训练命令.md').read_text(),re.S) if 'python train_pi_robo_async.py' in b)
args=shlex.split(block.replace('\\\n',' ')); args=args[args.index('train_pi_robo_async.py'):]
sys.argv=args
import train_pi_robo_async as t
import jax, numpy as np
from client.ur.policy import example_observation
from client.envs.ur_env import to_expo_observation
from expo_ft.utils.ur_rollout import prepare_rollout, prepare_update
from expo_ft.utils.jax_ready import block_until_ready_serial
if options.wait_mode == 'batched':
 # Diagnostic-only restoration of the old waits; never enabled by production.
 import expo_ft.agents.alg.expo_ft as algorithm
 algorithm.block_until_ready_serial = jax.block_until_ready
 block_until_ready_serial = jax.block_until_ready
from expo_ft.utils.train_utils import build_pi05_config
from openpi.training import config as oc
t.FLAGS(args); t.init_logging(); t.set_compilation_cache_dir('async-EXPOLearner')
assert t.FLAGS.checkpoint_interval==4000
_,cfg,_,_=build_pi05_config(t.FLAGS.config)
ref=oc.get_config('expo_pi05_droid_lora_finetune_sft_cartesian_state')
assert cfg.lr_schedule==ref.lr_schedule and cfg.optimizer==ref.optimizer
source=Path('train_pi_robo_async.py').read_text()
g=dict(vars(t)); g['resuming']=False
exec(textwrap.dedent(source[source.index('    num_gpus = jax.device_count()'):source.index('    log_dir = ')]),g)
setup=source[source.index('    if FLAGS.config_task.env_type in'):source.index('    start_step = 0')]
setup=setup.replace('capacity=FLAGS.max_steps,','capacity=4000,')
exec(compile(textwrap.dedent(setup),'async-entry-construction','exec'),g)
agent=g['agent']; bp=g['batch_processor']; rs=g['replicated_sharding']
print('SMOKE built real SFT2500 / 10 demos / batch64 UTD20; replay capacity only reduced to4000',flush=True)
t.diagnostics.configure(True, output, 'server')
prepare_rollout(agent,t.FLAGS.config,t.FLAGS.config_task,t.FLAGS.replan_steps,t.FLAGS.delay,str(output))
compiled=prepare_update(agent,bp,t.FLAGS.config_task,t.FLAGS.utd_ratio,rs)
print('SMOKE warmup finished',flush=True)
import h5py
import ctypes
profiler = ctypes.CDLL(str(root/'.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12')) if options.profile else None
with h5py.File('data/online_rl/success/10/traj.hdf5') as f:
 cached=[{k:f['saved_observation'][k][i] for k in ('state','exterior_image_1_left','exterior_image_2_left','wrist_image_left','prompt')} for i in range(0,320,10)]
for o in cached:
 o['prompt']=o['prompt'].decode() if isinstance(o['prompt'],bytes) else o['prompt']
obs=example_observation(); obs['prompt']=t.FLAGS.config_task.language_instruction
for k in ['observation/image','observation/wrist_image']: obs[k]=np.full((224,224,3),127,np.uint8)
obs=to_expo_observation(obs)
actor_agent=agent.cache_infer_params(); stop=threading.Event(); errors=[]; counts=[0]; published=[None]; lock=threading.Lock(); times=[]
heartbeats=[]; inference_times=[]
def heartbeat():
 last=time.perf_counter()
 while not stop.wait(0.02):
  now=time.perf_counter(); heartbeats.append((time.time(),now-last)); last=now
heart=threading.Thread(target=heartbeat,daemon=True); heart.start()
def infer():
 global actor_agent
 try:
  while not stop.is_set():
   with lock:
    if published[0] is not None:
     actor_agent=actor_agent.replace(_infer_cache=published[0]); published[0]=None
   t.diagnostics.set_context(step=counts[0])
   started=time.perf_counter()
   actions,actor_agent,_=actor_agent.sample_actions(cached[counts[0]%len(cached)])
   actions=np.asarray(jax.device_get(actions)); assert np.isfinite(actions).all() and actions.shape[1]==7
   inference_times.append(time.perf_counter()-started)
   counts[0]+=1
   stop.wait(0.8)
 except BaseException as e:
  errors.append(e); traceback.print_exc(); stop.set()
th=threading.Thread(target=infer); th.start()
try:
 learner=agent; rng=jax.random.PRNGKey(142)
 for i in range(options.updates):
  t.diagnostics.set_context(update=i+1)
  batch,ab,rng=bp.next_batch(rng)
  learner=learner.replace(rng=jax.device_put(learner.rng,rs))
  if profiler is not None and i == 0: profiler.cudaProfilerStart()
  start=time.perf_counter(); learner,info=learner.update(learner,batch,20,ab); block_until_ready_serial((learner,info))
  times.append(time.perf_counter()-start)
  if profiler is not None and i == 0: profiler.cudaProfilerStop()
  assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(info))
  with lock: published[0]=learner._infer_cache
  print('SMOKE update',i+1,'seconds',times[-1],'inferences',counts[0],flush=True)
  if errors: raise errors[0]
finally:
 stop.set(); th.join(); heart.join()
 (output/'results.json').write_text(json.dumps(dict(update_seconds=times,inference_seconds=inference_times,heartbeat=heartbeats,source=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),preallocate=os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),wait_mode=options.wait_mode,devices=[str(d) for d in jax.devices()]),indent=2))
if errors: raise errors[0]
print('SMOKE PASS',json.dumps(dict(update_seconds=times,inferences=counts[0],memory=[d.memory_stats() for d in jax.devices()])),flush=True)
