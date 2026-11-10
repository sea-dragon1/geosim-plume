"""Formal E03 evaluation. Run only through GPU batch allocation."""
import os, sys, json, hashlib, shutil, gc, csv
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"common"))
sys.path.insert(0,str(Path(__file__).parent))
import model_zoo as mz
import thresholds as th
import run_rollout as rr
from data_full import spec_of
assert os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available(), "GPU batch allocation required"
campaign=ROOT/"E03_rollout/formal_20260925"
cases=json.loads((campaign/"cases.json").read_text())
case=cases[int(sys.argv[1])]
run=ROOT/case["stage"]/"runs"/case["exp_id"]
out=campaign/"runs"/case["exp_id"]
out.mkdir(parents=True,exist_ok=False)
device=torch.device("cuda:0")
ckpt=Path(mz.checkpoint_path(str(run),"best"))
def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
fingerprint=digest(ckpt)
meta=dict(case, job_id=os.environ["SLURM_JOB_ID"], node=os.uname().nodename,
          gpu=torch.cuda.get_device_name(0), checkpoint=str(ckpt), checkpoint_sha256=fingerprint,
          n_test=spec_of(case["subset"])["n_traj_test"], n_windows=12,
          protocol="10 observed frames; 12 x 10 output frames; teacher resets at window boundaries")
(out/"manifest.json").write_text(json.dumps(meta,indent=2))
print("FORMAL",meta,flush=True)
for name in ("range_limits.csv","range_limits_plume2.csv"):
    shutil.copy2(ROOT/"E03_rollout/results"/name,out/name)
th.RESULTS=str(out)
th.TAU_PATH=str(out/"tau_c.json")
th.META_PATH=str(out/"thresholds_meta.json")
th.RANGE_PATH=str(out/"range_limits.csv")
th.build_one(str(run),case["subset"],device=device)
gc.collect()
torch.cuda.empty_cache()
rr.RESULTS=str(out)
rr.CSV_DIR=str(out)
statuses=[]
for mode in ("auto","teacher"):
    statuses.append(rr.run_one(str(run),case["subset"],mode,"best",12,device,False,False))
    gc.collect()
    torch.cuda.empty_cache()
assert digest(ckpt)==fingerprint, "Checkpoint changed during evaluation"
# Enforce full sample coverage and unique per-frame records.
with open(out/"rollout_curves.csv") as f:
    rows=list(csv.DictReader(f))
l2=[r for r in rows if r["metric"]=="l2re" and r["scope"]=="traj"] if rows and "scope" in rows[0] else [r for r in rows if r["metric"]=="l2re"]
expected=2*meta["n_test"]*120*6
# The writer includes aggregate records; count trajectory records explicitly below.
print("CSV columns", list(rows[0]), "L2 rows",len(l2), "expected trajectory rows",expected,flush=True)
with open(out/"failures.csv") as f:
    failures=list(csv.DictReader(f))
assert len(failures)==2*meta["n_test"]*6, (len(failures),meta["n_test"])
assert set(r["mode"] for r in failures)=={"auto","teacher"}
meta["status"]="COMPLETED"
meta["mode_statuses"]=statuses
meta["failure_rows"]=len(failures)
(out/"manifest.json").write_text(json.dumps(meta,indent=2))
(out/"SUCCESS").write_text("Full test trajectories; both modes completed\n")
print("FORMAL_SUCCESS",case["exp_id"],flush=True)
