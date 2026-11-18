import sys, warnings, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
warnings.filterwarnings("ignore")
cols=["age","workclass","fnlwgt","education","education_num","marital_status","occupation","relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country","income"]
df=pd.read_csv(sys.argv[1],header=None,names=cols,keep_default_na=False,na_values=[]); out=sys.argv[2]
df["y"]=(df.income==">50K").astype(int)
SURF,INK,INK2,MUTED,GRID="#fcfcfb","#0b0b0b","#52514e","#898883","#e6e5e1"; BLUE,ORANGE="#2a78d6","#eb6834"
RAMP=["#cde2fb","#9ec5f4","#6da7ec","#3987e5","#256abf","#184f95","#0d366b"]
plt.rcParams.update({"figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,"text.color":INK,"axes.labelcolor":INK2,
 "xtick.color":INK2,"ytick.color":INK2,"axes.edgecolor":GRID,"font.size":10,"axes.titlesize":11,"axes.titleweight":"bold","axes.titlelocation":"left"})
def clean(ax,grid="y"):
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.grid(axis=grid,color=GRID,lw=.7); ax.set_axisbelow(True); ax.tick_params(length=0)
# ---------- FIGURE 1: what is hiding in the raw columns ----------
fig,ax=plt.subplots(1,3,figsize=(15,4.6),gridspec_kw={"wspace":.28})
a=ax[0]; clean(a,"x")
q=pd.Series({c:(df[c]=="?").mean()*100 for c in ["workclass","occupation","native_country"]})
a.barh(q.index[::-1],q.values[::-1],color=BLUE,height=.5)
for i,(n,v) in enumerate(q[::-1].items()): a.text(v+.1,i,f"{v:.2f}%  ({int((df[n]=='?').sum()):,} rows)",va="center",color=INK2,fontsize=9)
a.set_xlim(0,9); a.set_title("Hidden missing values: '?' strings"); a.set_xlabel("% of 48,842 rows (pandas reports 0 NaN)")
b=ax[1]; clean(b)
nz=df.loc[df.capital_gain>0,"capital_gain"]; bins=np.logspace(np.log10(114),np.log10(100000),45)
b.hist(nz[nz<99999],bins=bins,color=BLUE,edgecolor=SURF,lw=.6); b.hist(nz[nz==99999],bins=[90000,100000],color=ORANGE,edgecolor=SURF,lw=.6)
b.set_xscale("log"); b.set_title("capital_gain, non-zero rows only (92% are 0)"); b.set_xlabel("capital_gain (log scale)"); b.set_ylabel("rows")
b.annotate("99,999:\n244 rows,\nall >50K",xy=(99999,250),xytext=(98000,520),ha="right",color=INK2,fontsize=9,arrowprops=dict(arrowstyle="-",color=MUTED,lw=.8))
c=ax[2]; clean(c); hp=df.hours_per_week.value_counts().sort_index()
c.bar(hp.index,hp.values,color=BLUE,width=.8); c.set_yscale("log"); c.set_title("hours_per_week: spikes at round numbers"); c.set_xlabel("hours per week"); c.set_ylabel("rows (log scale)")
c.text(40,hp[40]*1.25,"40: 22,803 rows",color=INK2,fontsize=9,ha="center")
c.text(100,1500,"99: 137 rows",color=INK2,fontsize=9,ha="right")
c.plot([99,99],[hp[99]*1.2,1100],color=MUTED,lw=.8)
c.set_ylim(.8,hp.max()*4)
fig.savefig(f"{out}/fig1_hidden_issues.png",dpi=150,bbox_inches="tight"); plt.close(fig)
# ---------- FIGURE 2: is '?' random? + signal + dependence ----------
fig,ax=plt.subplots(1,3,figsize=(15,4.8),gridspec_kw={"wspace":.3,"width_ratios":[1,1,1.25]})
a=ax[0]; clean(a); lab=["workclass","occupation","native_country"]; kn=[df.loc[df[c]!="?","y"].mean()*100 for c in lab]; mi=[df.loc[df[c]=="?","y"].mean()*100 for c in lab]
x=np.arange(3); w=.36
a.bar(x-w/2-.01,kn,w,color=BLUE,label="value known"); a.bar(x+w/2+.01,mi,w,color=ORANGE,label="value is '?'")
for xi,v in zip(x-w/2-.01,kn): a.text(xi,v+.6,f"{v:.1f}",ha="center",color=INK2,fontsize=9)
for xi,v in zip(x+w/2+.01,mi): a.text(xi,v+.6,f"{v:.1f}",ha="center",color=INK2,fontsize=9)
a.set_xticks(x); a.set_xticklabels(lab); a.set_ylim(0,32); a.set_ylabel("% earning >50K"); a.set_title("'?' is NOT random for job columns"); a.legend(frameon=False,loc="upper right",fontsize=9)
b=ax[1]; clean(b); df["age_bin"]=pd.cut(df.age,[16,25,35,45,55,65,91],labels=["17-25","26-35","36-45","46-55","56-65","66+"]); t=df.groupby("age_bin",observed=True).y.mean()*100
b.bar(t.index.astype(str),t.values,color=BLUE,width=.6)
for i,v in enumerate(t.values): b.text(i,v+.6,f"{v:.0f}%",ha="center",color=INK2,fontsize=9)
b.set_ylim(0,46); b.set_title("P(>50K) by age: not a straight line"); b.set_xlabel("age group"); b.set_ylabel("% earning >50K")
c=ax[2]; cc=["workclass","education","marital_status","occupation","relationship","race","sex","native_country","income"]
def cv(x,y):
    tb=pd.crosstab(x,y); chi=stats.chi2_contingency(tb,correction=False)[0]; n=tb.values.sum(); return np.sqrt(chi/(n*(min(tb.shape)-1)))
M=np.array([[cv(df[i],df[j]) for j in cc] for i in cc])
from matplotlib.colors import ListedColormap
im=c.imshow(M,cmap=ListedColormap(RAMP[:0]+RAMP) if False else matplotlib.colors.LinearSegmentedColormap.from_list("b",RAMP),vmin=0,vmax=1)
c.set_xticks(range(len(cc))); c.set_xticklabels(cc,rotation=45,ha="right"); c.set_yticks(range(len(cc))); c.set_yticklabels(cc)
for i in range(len(cc)):
    for j in range(len(cc)):
        if i!=j and M[i,j]>=.35: c.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",fontsize=8,color="white" if M[i,j]>.55 else INK)
c.set_title("Association between categorical columns (Cramér's V)"); c.tick_params(length=0)
for s in c.spines.values(): s.set_visible(False)
fig.colorbar(im,ax=c,fraction=.046,pad=.03).outline.set_visible(False)
fig.savefig(f"{out}/fig2_missingness_signal_dependence.png",dpi=150,bbox_inches="tight"); plt.close(fig)
print("ok")
