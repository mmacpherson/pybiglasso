import datetime,json,re,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent
start=time.monotonic(); quiet_since=None; next_sample=start
with (p/'gate.jsonl').open('a',buffering=1) as log, (p/'gate-top.txt').open('a',buffering=1) as raw:
    while True:
        time.sleep(max(0,next_sample-time.monotonic()))
        sample_start=datetime.datetime.now(datetime.timezone.utc).isoformat()
        output=subprocess.check_output(['top','-l','2','-n','0','-s','5'],text=True)
        raw.write('SAMPLE '+sample_start+'\n'+output+'\n')
        idle=float(re.findall(r'CPU usage:.*?([\d.]+)% idle',output)[-1])
        loads=[float(v) for v in re.findall(r'Load Avg:\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)',output)[-1]]
        now=time.monotonic();passed=idle>=90 and loads[0]<3
        quiet_since=(now if quiet_since is None else quiet_since) if passed else None
        quiet=0 if quiet_since is None else now-quiet_since
        r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=now-start,cpu_idle_pct=idle,load_1m=loads[0],load_5m=loads[1],load_15m=loads[2],qualifies=passed,consecutive_quiet_seconds=quiet)
        log.write(json.dumps(r)+'\n')
        if quiet>=180:
            (p/'gate-passed.json').write_text(json.dumps(r,indent=2)+'\n');print('PASSED '+json.dumps(r),flush=True);break
        if now-start>=1200:
            (p/'gate-timeout.json').write_text(json.dumps(r,indent=2)+'\n');print('TIMEOUT '+json.dumps(r),flush=True);break
        next_sample=min(next_sample+15,start+1200)
