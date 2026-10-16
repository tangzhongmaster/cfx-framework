import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
U="../"
OUT="./"
e10=pd.read_csv(U+"results_exp10.csv");e11=pd.read_csv(U+"results_exp11.csv")
C_BLUE="#0072B2";C_RED="#D55E00"
plt.rcParams.update({"font.size":10,"figure.dpi":150,"axes.spines.top":False,"axes.spines.right":False})
fig,axes=plt.subplots(2,1,figsize=(6.4,2.6),sharex=True)
for ax,d,label,color in [(axes[0],e11,"NAB",C_BLUE),(axes[1],e10,"MSL",C_RED)]:
    v=d[d.feasible==True].n_dims_changed.tolist()
    ax.scatter(v,[0]*len(v),s=55,color=color,alpha=0.6,zorder=3,edgecolor="none")
    ax.set_yticks([]);ax.set_xlim(0,52)
    ax.text(51,0.35,f"{label}: {len(v)}/{len(d)} feasible",ha="right",va="center",fontsize=8)
    ax.set_ylim(-0.5,0.6);ax.set_ylabel(label,rotation=0,ha="right",va="center",fontsize=9)
axes[1].set_xlabel(r"required window length $\ell^\star$")
plt.tight_layout();plt.savefig(OUT+"fig4_temporal_window.pdf");plt.savefig(OUT+"fig4_temporal_window.png")
