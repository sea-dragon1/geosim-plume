"""Exact-time evaluation from saved error curves; no extrapolation."""
from collections import defaultdict
import numpy as np
import h5py
from pathlib import Path

def interpolate_mean(curves, targets):
    values=[]
    for trajectory, points in curves.items():
        points=sorted(points)
        t=np.array([p[0] for p in points]);e=np.array([p[1] for p in points])
        assert np.all(np.diff(t)>0), "Nonunique timestamps"
        target=targets[trajectory]
        if t[0]<=target<=t[-1]:
            values.append(float(np.interp(target,t,e)))
    if not values:return float("nan"),float("nan"),0
    return float(np.mean(values)),float(np.std(values,ddof=1)) if len(values)>1 else 0.0,len(values)

def selfcheck():
    curves={0:[(11.,2.),(13.,6.),(16.,12.)],1:[(12.,4.),(15.,10.)]}
    m,sd,n=interpolate_mean(curves,{0:12.,1:13.5})
    assert n==2 and m==5.5
    m,sd,n=interpolate_mean(curves,{0:10.,1:16.})
    assert n==0 and np.isnan(m)
    m,sd,n=interpolate_mean(curves,{0:13.,1:15.})
    assert n==2 and m==8.

def recompute(curve_rows, summary_rows, spec_of):
    selfcheck()
    grouped=defaultdict(lambda:defaultdict(list))
    for row in curve_rows:
        if row["metric"]=="l2re":
            grouped[(row["run_id"],row["field"])][int(row["trajectory"])].append((float(row["physical_time"]),float(row["value"])))
    anchors={}
    for subset in {r["subset"] for r in summary_rows}:
        spec=spec_of(subset)
        with h5py.File(Path(spec["data_dir"])/spec["test_file"]) as f:
            anchors[subset]=np.asarray(f["t-coordinate"][:,9],dtype=float)/spec["t_divisor"]
    result=[]
    for row in summary_rows:
        curves=grouped[(row["run_id"],row["field"])]
        target=float(row["physical_time"])
        targets={i:float(anchors[row["subset"]][i])+target if row["endpoint_kind"]=="lead" else target for i in curves}
        mean,sd,n=interpolate_mean(curves,targets)
        new=dict(row,mean=mean,sd_across_traj=sd,n_valid=n,covered=int(n>0))
        result.append(new)
    return result
