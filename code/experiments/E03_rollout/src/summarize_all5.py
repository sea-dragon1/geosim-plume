"""Validate, merge and summarize formal E03 outputs on a compute node."""
import os, sys, json, csv, shutil
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"common"))
sys.path.insert(0,str(Path(__file__).parent))
import plot_rollout as pr
campaign=ROOT/"E03_rollout/formal_all5_20260925"
cases=json.loads((campaign/"cases.json").read_text())
merged=campaign/"merged"
merged.mkdir(exist_ok=True)
allrows={name:[] for name in ("rollout_curves","rollout_summary","failures")}
tau={}
metas=[]
for case in cases:
    out=ROOT/"E03_rollout"/case["source_campaign"]/"runs"/case["exp_id"]
    assert (out/"SUCCESS").exists(), "Incomplete: "+case["exp_id"]
    meta=json.loads((out/"manifest.json").read_text())
    metas.append(meta)
    tm=json.loads((out/"thresholds_meta.json").read_text())["runs"]
    assert len(tm)==1 and tm[0]["n_val_traj"]==(10 if case["subset"]=="plume1d" else 11), tm
    tau.update({k:v for k,v in json.loads((out/"tau_c.json").read_text()).items() if not k.startswith("_")})
    for name in allrows:
        with open(out/(name+".csv")) as f: rows=list(csv.DictReader(f))
        if name=="rollout_curves":
            assert len(rows)==2*meta["n_test"]*120*6*2, (case,len(rows))
            keys=[tuple(r[k] for k in ("mode","trajectory","frame","field","metric")) for r in rows]
            assert len(set(keys))==len(keys), "Duplicate prediction records"
            assert {r["mode"] for r in rows}=={"auto","teacher"}
            assert {int(r["trajectory"]) for r in rows}==set(range(meta["n_test"]))
            assert all(np.isfinite(float(r["value"])) for r in rows)
        allrows[name].extend(rows)
from endpoint_summary import recompute
allrows["rollout_summary"]=recompute(allrows["rollout_curves"],allrows["rollout_summary"],pr.spec_of)
for name, rows in allrows.items():
    with open(merged/(name+".csv"),"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
(merged/"tau_c.json").write_text(json.dumps(tau,indent=2))
(merged/"manifests.json").write_text(json.dumps(metas,indent=2))
# First autonomous window must reproduce the E01/E02 short-window score.
refs={}
for stage, filename in (("E01_main","E01_main_eq9_external.csv"),("E02_ablation","E02_ablation_eq9.csv")):
    with open(ROOT/stage/"results"/filename) as f:
        for r in csv.DictReader(f): refs[r["exp_id"]]=r
first=defaultdict(list)
last=defaultdict(list)
for r in allrows["rollout_curves"]:
    if r["metric"]!="l2re": continue
    exp=r["run_id"].split("|")[0]
    if r["mode"]=="auto" and int(r["window"])==0: first[(exp,r["field"])].append(float(r["value"]))
    if int(r["window"])==11: last[(r["subset"],r["model"],r["mode"],r["seed"],r["field"])].append(float(r["value"]))
checks=[]
for (exp,field), values in first.items():
    actual=float(np.mean(values)); expected=float(refs[exp]["E_"+field])
    relative=abs(actual-expected)/max(abs(expected),1e-12)
    checks.append(dict(exp_id=exp,field=field,actual=actual,reference=expected,relative_difference=relative))
(merged/"first_window_check.json").write_text(json.dumps(checks,indent=2))
assert max(c["relative_difference"] for c in checks)<0.01, "First-window score mismatch >1%; inspect before publication"
assert not any(int(r["n_invalid"]) for r in allrows["failures"]), "Nonfinite predictions: review metric sanitation before reporting"
figdir=campaign/"figures"; figdir.mkdir(exist_ok=True)
curves=pr.load_curves(str(merged/"rollout_curves.csv"))
for subset in ("plume1d","plume2"):
    pr.plot_subset({k:v for k,v in curves.items() if k[0]==subset},subset,str(figdir/("rollout_curves_"+subset+".pdf")),modes=["auto","teacher"])
lines=["# E03 formal evaluation (2026-09-25)","",
       "20 checkpoints; both modes; full held-out test trajectories; 12 windows / 120 predicted frames.",
       "GeoSim-Plume1: 3 seeds, 12 trajectories. GeoSim-Plume2: seed0, 25 trajectories.",
       "Models: MFGNO, Independent MPNN, FNO, Transolver, GNOT.",
       "Teacher forcing resets true history once per 10-frame window for all models.",
       "GNNs repeat the trained 21-frame operator with legacy timestamp clamping at each window boundary. This is checkpoint compatibility, not corrected physical-time integration. Absolute timestamp conditioning is retained; one additional scheduled timestamp is required.",
       "Failure thresholds use validation rows of the training file (10 / 11 trajectories), never test rows.",
       "All computation ran in GPU batch allocations. Per-run manifests retain checkpoint SHA256 and hardware.",
       "First autonomous window vs E01/E02: maximum relative score difference {:.3g}.".format(max(c["relative_difference"] for c in checks)),
       "No nonfinite predictions. Failure categories are protocol violations, not automatically numerical blow-up.","",
       "## Last output window: field-wise relative L2","",
       "Seed means below average each seed's trajectory/frame mean; these are final-window values, not fixed-time endpoints.",
       "| Subset | Model | Mode | ro | pr | tk | nu | vx | vy |",
       "|---|---|---|---|---|---|---|---|---|"]
group=defaultdict(dict)
for (subset,model,mode,seed,field),values in last.items():
    group[(subset,model,mode)].setdefault(field,[]).append(float(np.mean(values)))
for key,fields in sorted(group.items()):
    lines.append("| "+" | ".join(key)+" | "+" | ".join("{:.5g}".format(np.mean(fields[f])) for f in ("ro","pr","tk","nu","vx","vy"))+" |")
lines+=["","## Failure records (trajectory-field pairs across seeds)","",
        "| Subset | Model | Mode | none | over_tau | range | nan_inf |","|---|---|---|---|---|---|---|"]
fc=defaultdict(Counter)
for r in allrows["failures"]: fc[(r["subset"],r["model"],r["mode"])][r["fail_kind"]]+=1
for key,counts in sorted(fc.items()):
    lines.append("| "+" | ".join(key)+" | "+" | ".join(str(counts[k]) for k in ("none","over_tau","range","numerical"))+" |")
lines+=["","Fixed 2/5/8/12 Myr lead endpoints are measured from the last observed input frame (index 9), by linear interpolation of per-frame relative errors within each trajectory support only; no extrapolation. Coverage is in merged/rollout_summary.csv. Older nearest-window summaries are superseded.",
        "Per-field curves with seed and trajectory variation are in figures/; each PDF has a manifest.",
        "Old smoke and selfcheck outputs are excluded. Earlier external thresholds were invalid and are not reused.",""]
(campaign/"REPORT.md").write_text("\n".join(lines))

print("\n".join(lines))
exec(compile((ROOT/"E03_rollout/src/plot_all5.py").read_text(), "plot_all5.py", "exec"))
(campaign/"SUCCESS").write_text("All cases, coverage, first-window checks, nonfinite checks, and plots passed\n")
print("FORMAL_CAMPAIGN_COMPLETE")
