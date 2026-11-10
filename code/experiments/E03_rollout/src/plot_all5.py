# Executed by summarize_all5.py after all acceptance checks pass.
import matplotlib.pyplot as plt
from collections import defaultdict
# ⚠ 不设这一行，PDF 里全是 Type 3 字体，部分阅读器/投稿系统打不开（同 make_fig01.py）
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
fig,axes=plt.subplots(3,4,figsize=(14,9),squeeze=False)
fields=("ro","pr","tk","nu","vx","vy")
models=("fno","GNOT","MPNN","Transolver","MFGNO")
manifest={"mode":"auto","note":"Light band: trajectory SD of first seed. Dark band: seed SD. GNN: legacy 21-frame window operator.","panels":[]}
for si,subset in enumerate(("plume1d","plume2")):
    for fi,field in enumerate(fields):
        ax=axes[fi//2,si*2+fi%2]
        for model in models:
            seeds=sorted({key[2] for key in curves if key[0]==subset and key[1]==model and key[3]=="auto"})
            vals=[]
            for seed in seeds:
                grid,mat,n=pr._interp_common(curves[(subset,model,seed,"auto")][field])
                assert grid is not None
                if vals: np.testing.assert_allclose(grid,refgrid,rtol=0,atol=1e-10)
                refgrid=grid
                vals.append(mat.mean(axis=0))
                if seed==seeds[0]:
                    mu,sd=mat.mean(axis=0),mat.std(axis=0,ddof=1)
                    ax.fill_between(grid,np.maximum(mu-sd,1e-12),mu+sd,color=pr.MODEL_COLORS[model],alpha=.09,linewidth=0)
            arr=np.asarray(vals)
            mu=arr.mean(axis=0)
            if len(seeds)>1:
                sd=arr.std(axis=0,ddof=1)
                ax.fill_between(grid,np.maximum(mu-sd,1e-12),mu+sd,color=pr.MODEL_COLORS[model],alpha=.24,linewidth=0)
            ax.plot(grid,mu,label=pr.MODEL_LABEL[model],color=pr.MODEL_COLORS[model],lw=1.1)
            manifest["panels"].append(dict(subset=subset,field=field,model=model,seeds=seeds,n_trajectories=n,t_min=float(grid[0]),t_max=float(grid[-1])))
        ax.axvline(float(pr.spec_of(subset)["t_train"]),color="black",ls="--",lw=.7)
        ax.set_yscale("log")
        ax.set_title(subset+" / "+pr.FLABEL[field])
        ax.set_xlabel("Physical time (Myr)")
        ax.set_ylabel("Relative L2 error")
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc="upper center",ncol=5,frameon=False)
fig.suptitle("Autonomous rollout: 12 windows / 120 frames",y=.995)
fig.tight_layout(rect=(0,0,1,.94))
fig.savefig(figdir/"fig04_rollout.pdf")
plt.close(fig)
(figdir/"fig04_rollout_manifest.json").write_text(json.dumps(manifest,indent=2))
# Preserve exact fixed-lead endpoints with seed mean and standard deviation.
groups=defaultdict(list)
for r in allrows["rollout_summary"]:
    if r["endpoint_kind"]=="lead" and r["metric"]=="l2re" and int(r["n_valid"])>0:
        groups[(r["subset"],r["model"],r["mode"],r["endpoint_label"],r["field"])].append(r)
endpoint_rows=[]
for key,rows in sorted(groups.items()):
    values=np.array([float(r["mean"]) for r in rows])
    endpoint_rows.append(dict(zip(("subset","model","mode","endpoint","field"),key),
          n_seeds=len(rows),mean=float(values.mean()),
          seed_sd=float(values.std(ddof=1)) if len(rows)>1 else "",
          min_n_valid=min(int(r["n_valid"]) for r in rows)))
with open(merged/"endpoint_seed_summary.csv","w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(endpoint_rows[0]));writer.writeheader();writer.writerows(endpoint_rows)
print("COMBINED_F04_AND_ENDPOINTS_COMPLETE")
