import datetime,json,os,re,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent
assert (p/'gate-passed.json').exists(),'Gate has not passed'
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
def snapshot(label):
    return dict(label=label,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),uptime=subprocess.check_output(['uptime'],text=True).strip(),load=subprocess.check_output(['sysctl','-n','vm.loadavg'],text=True).strip(),power=re.sub(r'\(id=\d+\)','(battery identifier redacted)',subprocess.check_output(['pmset','-g','batt'],text=True).strip()),thermal=subprocess.check_output(['pmset','-g','therm'],text=True).strip())
with (p/'set-environment.jsonl').open('a',buffering=1) as log:
    for family,dtype,problems,alphas in [('binomial','int8',['wide-2k-20k','wide-5k-50k'],['1','.5']),('binomial','float64',['tall-5k-1k','wide-2k-20k'],['1','.5']),('gaussian','float64',['tall-5k-1k','wide-2k-20k'],['.5'])]:
        for problem in problems:
            for alpha in alphas:
                label=f'{family}-{dtype}-{problem}-alpha{alpha}'
                command=[str(p/'gated/.venv/bin/python'),str(p/'int8-threeway.py'),'--baseline-extension',str(p/'baseline/src/pybiglasso/_rust.abi3.so'),'--ungated-extension',str(p.parent/'candidate/src/pybiglasso/_rust.abi3.so'),'--family',family,'--dtype',dtype,'--problem',problem,'--alpha',alpha,'--threads','1','8','--repeats','6','--out',str(p/f'{family}-{dtype}.jsonl')]
                log.write(json.dumps(snapshot(label+'-before'))+'\n')
                with (p/(label+'.stdout')).open('w') as stdout:
                    subprocess.run(command,cwd=p/'gated',env=env,stdout=stdout,check=True)
                log.write(json.dumps(snapshot(label+'-after'))+'\n')
                print('COMPLETED '+label,flush=True)
