import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
e7=pd.read_csv("../results_exp7.csv")
OUT="./"
C_BLUE="#0072B2";C_ORANGE="#E69F00";C_GREEN="#009E73"
plt.rcParams.update({"font.size":10,"figure.dpi":150,"axes.spines.top":False,"axes.spines.right":False})
ec=e7[e7.excluded!=True];rows=[]
for eps,g in ec.groupby("epsilon"):
    f=g[g.feasible==True];rows.append(dict(epsilon=eps,feas_pct=100*len(f)/len(g),dyhat=f.abs_delta_y_hat.median(),dw=f.abs_delta_w.median()))
t=pd.DataFrame(rows).sort_values("epsilon")
xl=[r"$10^{-4}$",r"$10^{-2}$",r"$5\times10^{-2}$",r"$10^{-1}$"]
fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
ax[0].plot(range(4),t.feas_pct,"o-",color=C_BLUE);ax[0].set_xticks(range(4));ax[0].set_xticklabels(xl)
ax[0].set_ylabel("feasibility (%)");ax[0].set_xlabel(r"$\epsilon$");ax[0].set_title("(a) Feasibility")
ax[1].plot(range(4),t.dyhat,"o-",color=C_ORANGE,label=r"$|\Delta\hat y|$");ax[1].plot(range(4),t.dw,"s--",color=C_GREEN,label=r"$|\Delta w|$")
ax[1].set_xticks(range(4));ax[1].set_xticklabels(xl);ax[1].set_xlabel(r"$\epsilon$");ax[1].set_title("(b) Correction (median, feasible)")
ax[1].legend(fontsize=8,frameon=False);plt.tight_layout()
plt.savefig(OUT+"fig3_epsilon.pdf");plt.savefig(OUT+"fig3_epsilon.png")
