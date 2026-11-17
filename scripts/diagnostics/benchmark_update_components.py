import os, sys, re, shlex, time, textwrap, json
from pathlib import Path
root = Path(__file__).resolve().parents[2]
os.chdir(root)
sys.path.insert(0,str(root))
command = next(b for b in re.findall(r'```bash\n(.*?)```',(root/'expo-setup/训练命令.md').read_text(),re.S) if 'python train_pi_robo.py' in b)
parts=shlex.split(command.replace('\\\n',' ')); sys.argv=parts[parts.index('train_pi_robo.py'):]
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
assert jax.device_count()==4
mesh=t.openpi_sharding.make_mesh(t.FLAGS.fsdp_devices)
ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(t.openpi_sharding.DATA_AXIS))
rs=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
t.set_compilation_cache_dir('server-audit-standard')
# Execute the entry point's exact dataset/model/buffer/agent construction, excluding
# W&B, checkpoint output and network/robot initialization. Only replay capacity is
# reduced for this one-update diagnostic; model/batch/UTD remain the documented defaults.
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
results = {}

def measure(name, fn, *args):
    # Warm exactly this compiled signature, discard output, then time ONE call.
    st=time.perf_counter()
    out=fn(*args)
    jax.block_until_ready(out)
    stage(name+' compile/warmup '+str(time.perf_counter()-st))
    del out
    gc.collect()
    st=time.perf_counter()
    out=fn(*args)
    jax.block_until_ready(out)
    seconds=time.perf_counter()-st
    results[name]=seconds
    stage(name+' ONE call seconds='+str(seconds))
    del out
    gc.collect()

@jax.jit
def prepare(agent,batch):
    batch=batch.copy()
    rng,k1=jax.random.split(agent.rng)
    rng,k2=jax.random.split(rng)
    batch['image']=agent.data_augmentation_fn(k1,batch['image'])
    batch['next_image']=agent.data_augmentation_fn(k2,batch['next_image'])
    batch=prepare_critic_batch(batch,agent.actor.model_config.action_dim,agent.action_dim,agent.state_dim,agent.action_horizon,agent.replan_steps,agent.critic_camera_keys)
    return agent.replace(rng=rng),batch

prepared,last=prepare(agent,batch)
jax.block_until_ready((prepared,last))
critic_one=jax.jit(lambda a,b:a.update_critic(b))
measure('critic_one_batch64_including_pi05_target_inference',critic_one,prepared,last)
critic_one.clear_cache()

@jax.jit
def pi05_one(a,b):
    b=b.copy()
    rng,key=jax.random.split(a.rng)
    b['image']=a.data_augmentation_fn(key,b['image'])
    b=prepare_critic_batch(b,a.actor.model_config.action_dim,a.action_dim,a.state_dim,a.action_horizon,a.replan_steps,a.critic_camera_keys)
    return a.replace(rng=rng).update_actor(b)
assert agent.actor_success_only, 'diagnostic expects configured success-only Pi05 batch'
measure('pi05_one_batch64',pi05_one,prepared,actor_batch)
pi05_one.clear_cache()
edit_one=jax.jit(lambda a,b:a.update_edit_actor(b))
measure('residual_actor_one_batch64',edit_one,prepared,last)
edit_one.clear_cache()
temp_one=jax.jit(lambda a:a.update_temperature(jax.numpy.array(0.,dtype=jax.numpy.float32)))
measure('temperature_one',temp_one,prepared)
temp_one.clear_cache()
del prepared,last,batch,actor_batch
prepare.clear_cache()
gc.collect()

# Keep preprocessing inside the 20-step executable, as production does: returning
# all 1280 stacked images would materialize otherwise fused buffers and inflate RAM.
batch,actor_batch=bp.to_device(jax.tree.map(lambda x:x,host_batch),jax.tree.map(lambda x:x,host_actor))
jax.block_until_ready((batch,actor_batch))
@jax.jit
def critic_twenty(a,b):
    a,b=prepare(a,b)
    mini=jax.tree.map(lambda x:x.reshape((t.FLAGS.utd_ratio,-1)+x.shape[1:]),b)
    mini=jax.tree.map(lambda x:jax.device_put(x,jax.sharding.NamedSharding(ds.mesh,jax.sharding.PartitionSpec(None,t.openpi_sharding.DATA_AXIS,*([None]*(x.ndim-2))))),mini)
    return jax.lax.scan(lambda a,b:a.update_critic(b),a,mini)
measure('critic_20_scan_with_preprocessing',critic_twenty,agent,batch)
critic_twenty.clear_cache()
prepare.clear_cache()
gc.collect()

# Unmodified fused production update: warm initial/steady signatures, then three
# sequential calls (the configured episode update count). Upload is synchronized.
for i in range(3):
    st=time.perf_counter()
    agent,info=agent.update(agent,batch,t.FLAGS.utd_ratio,actor_batch)
    jax.block_until_ready((agent,info))
    stage('fused warmup '+str(i+1)+' seconds='+str(time.perf_counter()-st))
full=[]
for i in range(t.FLAGS.num_updates):
    st=time.perf_counter()
    agent,info=agent.update(agent,batch,t.FLAGS.utd_ratio,actor_batch)
    jax.block_until_ready((agent,info))
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(info))
    full.append(time.perf_counter()-st)
    stage('fused ONE update '+str(i+1)+' seconds='+str(full[-1]))
results.update(fused_update_seconds=full,episode_update_total_seconds=sum(full),batch_size=t.FLAGS.batch_size,utd=t.FLAGS.utd_ratio,num_updates=t.FLAGS.num_updates,devices=[str(d) for d in jax.devices()])
print('RESULT '+json.dumps(results),flush=True)
