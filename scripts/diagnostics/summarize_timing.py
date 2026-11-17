"""Summarize separate timing JSONL files; milliseconds, not W&B metrics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('paths',nargs='+',type=Path)
p.add_argument('--slow-ms',type=float,default=500)
a=p.parse_args()
stats=defaultdict(list);slow=[];active={};dropped={}
for path in a.paths:
    for line in path.open():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue  # Running writer may have a partial last line.
        key=(str(path),r['pid'],r['id'])
        if r['event']=='begin':active[key]=r
        elif r['event']=='end':
            active.pop(key,None)
            stats[(str(path),r['span'])].append(r['duration_ms'])
            if r['duration_ms']>=a.slow_ms:slow.append(dict(file=str(path),**r))
        dropped[str(path)]=max(dropped.get(str(path),0),r.get('dropped_events',0))
print('file,span,count,p50_ms,p95_ms,p99_ms,max_ms')
for (path,name),values in sorted(stats.items()):
    q=np.percentile(values,[50,95,99,100]);print(f'{path},{name},{len(values)},'+','.join(f'{v:.3f}' for v in q))
print('SLOW_EVENTS (nested spans overlap; do not sum):')
for r in sorted(slow,key=lambda r:r['wall_ns']):print(json.dumps(r))
print('UNMATCHED_BEGINS (in progress, crash or dropped event):',json.dumps(list(active.values())))
print('DROPPED_EVENTS:',json.dumps(dropped))
