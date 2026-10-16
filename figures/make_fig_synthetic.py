import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
U="../"
OUT="./"
e13=pd.read_csv(U+"results_exp13.csv");e14=pd.read_csv(U+"results_exp14.csv")
C_BLUE="#0072B2";C_ORANGE="#E69F00"
plt.rcParams.update({"font.size":10,"figure.dpi":150,"axes.spines.top":False})
def tab(d,k):
    d=d[d.excluded!=True];r=[]
    for v,g in d.groupby(k):
        f=g[g.feasible==True];r.append((v,f.distance.median(),f.abs_delta_w.median()))
    return pd.DataFrame(r,columns=["x","dist","dw"]).sort_values("x")
t13=tab(e13,"anomaly_magnitude");t14=tab(e14,"d_regime")
fig,axes=plt.subplots(1,2,figsize=(7.2,2.8))
for ax,t,xl,ti in [(axes[0],t13,r"anomaly magnitude ($\sigma$)","(a) Anomaly magnitude"),(axes[1],t14,r"$D_{\mathrm{regime}}$","(b) Regime discrepancy")]:
    ax.plot(t.x,t.dist,"o-",color=C_BLUE);ax2=ax.twinx();ax2.plot(t.x,t.dw,"s--",color=C_ORANGE)
    ax.set_xlabel(xl);ax.set_ylabel("median distance",color=C_BLUE);ax2.set_ylabel(r"median $|\Delta w|$",color=C_ORANGE);ax.set_title(ti)
plt.tight_layout();plt.savefig(OUT+"fig5_synthetic.pdf");plt.savefig(OUT+"fig5_synthetic.png")
