import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
U="../"
OUT="./"
e6=pd.read_csv(U+"results_exp6.csv")
C_BLUE="#0072B2";C_RED="#D55E00";C_GREY="#555555"
plt.rcParams.update({"font.size":10,"figure.dpi":150,"axes.spines.top":False,"axes.spines.right":False})
fig,axes=plt.subplots(2,2,figsize=(6.4,5.6))
for row,bb in enumerate(["raw_context","spci"]):
    for col,ds in enumerate(["msl","nab"]):
        ax=axes[row,col]
        s=e6[(e6.excluded!=True)&(e6.dataset==ds)&(e6.backbone==bb)&(e6.feasible==True)]
        x=s.m0.abs().values;y=s.abs_delta_y_hat.values
        ax.scatter(x,y,s=18,color=C_BLUE,alpha=0.65,edgecolor="none")
        a,b=np.polyfit(x,y,1);r2=1-((y-(a*x+b))**2).sum()/((y-y.mean())**2).sum()
        xs=np.linspace(0,x.max(),50)
        ax.plot(xs,a*xs+b,color=C_RED,lw=1.6,label=f"fit: a={a:.2f}, $R^2$={r2:.2f}")
        ax.plot(xs,xs,"--",color=C_GREY,lw=1,label="y=x")
        ax.set_title(f"{bb}, {ds.upper()}",fontsize=9)
        if row==1: ax.set_xlabel(r"$|m_0|$")
        if col==0: ax.set_ylabel(r"$|\Delta\hat y|$")
        ax.legend(fontsize=6.5,frameon=False,loc="upper left")
plt.tight_layout()
plt.savefig(OUT+"fig2_center_shift.pdf");plt.savefig(OUT+"fig2_center_shift.png")
