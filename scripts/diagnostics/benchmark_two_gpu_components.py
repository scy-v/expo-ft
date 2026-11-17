"""Measure isolated EXPO updates on two GPUs, without a robot or training writes."""
import os, sys, re, shlex, time, textwrap, json
from pathlib import Path
root = Path(__file__).resolve().parents[2]
os.chdir(root)
sys.path.insert(0,str(root))
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--batch-size',type=int,choices=[64,128],required=True)
parser.add_argument('--component',choices=['critic','edit_actor','pi05'],required=True)
parser.add_argument('--repeats',type=int,default=20)
parser.add_argument('--warmup',type=int,default=5)
parser.add_argument('--output',required=True)
opts=parser.parse_args()
if opts.repeats < 1 or opts.warmup < 1:
    parser.error('--repeats and --warmup must be positive')
command = next(b for b in re.findall(r'```bash\n(.*?)```',(root/'expo-setup/训练命令.md').read_text(),re.S) if 'python train_pi_robo.py' in b)
parts=shlex.split(command.replace('\\\n',' ')); sys.argv=parts[parts.index('train_pi_robo.py'):]
sys.argv=[a for a in sys.argv if not a.startswith(('--batch_size=', '--utd_ratio='))]+[f'--batch_size={opts.batch_size}', '--utd_ratio=1']
import train_pi_robo as t
import jax, numpy as np
import logging
logging.basicConfig(level=logging.INFO)
from expo_ft.utils.train_utils import build_pi05_config
from openpi.training import config as oc
from configs.model.expo_ft_pi_config import get_config
start=time.time()
def stage(message): print(f'AUDIT {time.time()-start:.1f}s {message}',flush=True)
t.FLAGS(sys.argv)
_,cfg,_,_=build_pi05_config(t.FLAGS.config)
base=oc.get_config('expo_pi05_droid_lora_finetune_sft_cartesian_state')
assert cfg.lr_schedule==base.lr_schedule and cfg.optimizer==base.optimizer
stage(f'optimizer and LR match author: {cfg.lr_schedule}; devices={jax.devices()}')
assert jax.device_count()==2
mesh=t.openpi_sharding.make_mesh(t.FLAGS.fsdp_devices)
ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(t.openpi_sharding.DATA_AXIS))
rs=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
t.set_compilation_cache_dir('components-two-gpu')
# Execute the entry point's exact dataset/model/buffer/agent construction, excluding
# W&B, checkpoint output and network/robot initialization. Only replay capacity is
# reduced for this diagnostic. Batch is selected by CLI; UTD=1 only sizes the
# sampled input for a single component update, not the production training loop.
source=Path('train_pi_robo.py').read_text()
block=source[source.index('    if FLAGS.config_task.env_type in'):source.index('    start_step = 0')]
block=block.replace('capacity=FLAGS.max_steps,','capacity=4000,')
g=dict(vars(t)); g.update(mesh=mesh,data_sharding=ds,replicated_sharding=rs,resuming=False)
stage('building real 2500-step model and loading 10 demos')
exec(compile(textwrap.dedent(block),'train_pi_robo.py:audit-construction','exec'),g)
agent=g.pop('agent'); bp=g['batch_processor']
g.pop('actor_train_state', None); g.pop('target_actor_params', None)

# Isolated diagnostic: no robot connection, no formal checkpoints or replay writes.
import gc
from expo_ft.agents.alg.expo_ft import prepare_critic_batch
agent = agent.replace(_infer_cache=None, rng=jax.device_put(agent.rng,rs))
host_batch, host_actor, _ = bp.next_batch(jax.random.PRNGKey(42), to_device=False)
batch, actor_batch = bp.to_device(jax.tree.map(lambda x:x[-t.FLAGS.batch_size:],host_batch),jax.tree.map(lambda x:x,host_actor))
jax.block_until_ready((agent,batch,actor_batch))
@jax.jit
def prepare(agent,batch):
    batch=batch.copy()
    rng,k1=jax.random.split(agent.rng)
    rng,k2=jax.random.split(rng)
    batch['image']=agent.data_augmentation_fn(k1,batch['image'])
    batch['next_image']=agent.data_augmentation_fn(k2,batch['next_image'])
    batch=prepare_critic_batch(batch,agent.actor.model_config.action_dim,agent.action_dim,agent.state_dim,agent.action_horizon,agent.replan_steps,agent.critic_camera_keys)
    return agent.replace(rng=rng),batch


if opts.component=='pi05':
    b=actor_batch.copy()
    rng,key=jax.random.split(agent.rng)
    b['image']=agent.data_augmentation_fn(key,b['image'])
    prepared_batch=prepare_critic_batch(b,agent.actor.model_config.action_dim,agent.action_dim,agent.state_dim,agent.action_horizon,agent.replan_steps,agent.critic_camera_keys)
    fn=jax.jit(lambda a,b:a.update_actor(b))
else:
    agent,prepared_batch=prepare(agent,batch)
    fn=jax.jit(lambda a,b:a.update_critic(b)) if opts.component=='critic' else jax.jit(lambda a,b:a.update_edit_actor(b))
jax.block_until_ready((agent,prepared_batch))
assert prepared_batch['actions'].shape[0] == opts.batch_size
del batch,actor_batch,host_batch,host_actor
samples=[]
for i in range(opts.warmup+opts.repeats):
    agent=agent.replace(rng=jax.device_put(agent.rng,rs))
    before=fn._cache_size()
    st=time.perf_counter()
    new_agent,info=fn(agent,prepared_batch)
    jax.block_until_ready((new_agent,info))
    elapsed=time.perf_counter()-st
    if i>=opts.warmup:
        if fn._cache_size()!=before: raise RuntimeError('Compilation occurred during timed iterations')
        samples.append(elapsed)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(info))
    agent=new_agent
    stage(f'{opts.component} iteration {i+1} seconds={elapsed:.6f} warmup={i<opts.warmup}')
a=np.array(samples)
result=dict(component=opts.component,batch_size=opts.batch_size,per_gpu=opts.batch_size//2,
 repeats=opts.repeats,warmup=opts.warmup,seconds=samples,
 statistics=dict(min=float(a.min()),max=float(a.max()),mean=float(a.mean()),median=float(np.median(a)),p10=float(np.percentile(a,10)),p25=float(np.percentile(a,25)),p75=float(np.percentile(a,75)),p90=float(np.percentile(a,90))),
 devices=[str(d) for d in jax.devices()],memory=[d.memory_stats() for d in jax.devices()],
 allocator=dict(preallocate=os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),fraction=os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')),
 scope='One component update including forward/backward/optimizer and device communication; critic includes Pi05 target inference. Preprocessing, data upload, compilation, inference-cache publication and robot excluded. Reuses batch; parameters and RNG advance.',
 source=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip())
Path(opts.output).parent.mkdir(parents=True, exist_ok=True)
Path(opts.output).write_text(json.dumps(result,indent=2)+'\n')
print('RESULT '+json.dumps(result),flush=True)
