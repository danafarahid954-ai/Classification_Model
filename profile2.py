import sys, warnings, pandas as pd, numpy as np
from scipy import stats
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 200)
cols = ["age","workclass","fnlwgt","education","education_num","marital_status","occupation","relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country","income"]
df = pd.read_csv(sys.argv[1], header=None, names=cols, keep_default_na=False, na_values=[])
df["y"] = (df.income==">50K").astype(int)
cat = [c for c in cols if df[c].dtype!="int64" and c!="income"]; num = ["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]
print("== A. DUPLICATE DETAILS ==")
dup = df[df.duplicated(cols[:-1]+["income"], keep=False)]
print("rows involved in exact-duplicate groups:", len(dup), "| groups:", dup.groupby(cols).ngroups)
print("income rate among duplicated rows:", dup.y.mean().round(3), "vs overall", df.y.mean().round(3))
conf = df.groupby(cols[:-1])["y"].agg(["nunique","size"]); conf = conf[conf["nunique"]>1]; print("conflicting-label groups:\n", conf.to_string())
print("\n== B. LOGICAL CONSISTENCY ==")
m = df.groupby("education")["education_num"].nunique(); print("education -> education_num is 1-to-1:", (m==1).all(), "| num of education_num values:", df.education_num.nunique(), "vs education levels:", df.education.nunique())
print("Husband & Female:", ((df.relationship=="Husband")&(df.sex=="Female")).sum(), "| Wife & Male:", ((df.relationship=="Wife")&(df.sex=="Male")).sum())
print("Married-civ-spouse but relationship not Husband/Wife:", ((df.marital_status=="Married-civ-spouse")&~df.relationship.isin(["Husband","Wife"])).sum())
print("Husband/Wife but not married status:", (df.relationship.isin(["Husband","Wife"])&~df.marital_status.str.startswith("Married")).sum())
print("Never-worked / Without-pay with occupation known:", ((df.workclass.isin(["Never-worked","Without-pay"]))&(df.occupation!="?")).sum(), "of", df.workclass.isin(["Never-worked","Without-pay"]).sum())
print("Never-worked income>50K:", df.loc[df.workclass=="Never-worked","y"].sum(), "| Without-pay income>50K:", df.loc[df.workclass=="Without-pay","y"].sum())
print("age<=20 with education_num>=13 (bachelor+):", ((df.age<=20)&(df.education_num>=13)).sum(), "| age<=22 Doctorate/Prof-school:", ((df.age<=22)&df.education.isin(["Doctorate","Prof-school"])).sum())
print("hours_per_week>=80 and age>=75:", ((df.hours_per_week>=80)&(df.age>=75)).sum())
print("age==90 rows (top-coded?), income>50K rate:", df.loc[df.age==90,"y"].mean().round(3), "| count age 85..89:", df.age.between(85,89).sum())
print("\n== C. ROW ORDER / ORIGIN ==")
a, b = df.iloc[:32561], df.iloc[32561:]; print("income>50K rate first 32,561 rows:", a.y.mean().round(4), "| rest:", b.y.mean().round(4), "| '?' in first/rest:", (a[cat]=='?').any(axis=1).mean().round(4), (b[cat]=='?').any(axis=1).mean().round(4))
print("income>50K rate by 10 equal row blocks:", [round(x,3) for x in df.groupby(np.arange(len(df))//(len(df)//10+1)).y.mean()])
print("\n== D. OUTLIERS (flagged, NOT removed) ==")
rows=[]
for c in num:
    x=df[c]; q1,q3=x.quantile([.25,.75]); i=q3-q1; lo,hi=q1-1.5*i,q3+1.5*i; z=(x-x.mean())/x.std()
    rows.append([c,(x<lo).sum(),(x>hi).sum(),round(100*((x<lo)|(x>hi)).mean(),2),(z.abs()>3).sum(), x.max(), round(df.loc[(x>hi)|(x<lo),"y"].mean(),3) if ((x>hi)|(x<lo)).any() else np.nan])
print(pd.DataFrame(rows,columns=["col","IQR_low","IQR_high","pct_flagged","|z|>3","max","P(>50K | flagged)"]).to_string(index=False))
print("note: for capital_gain/loss IQR=0 so every non-zero is 'outlier'; use the non-zero subset instead:")
for c in ["capital_gain","capital_loss"]:
    nz=df.loc[df[c]>0,c]; print(f" {c}: n_nonzero={len(nz)} median={nz.median():.0f} p95={nz.quantile(.95):.0f} max={nz.max()} P(>50K|nonzero)={df.loc[df[c]>0,'y'].mean():.3f} (zero: {df.loc[df[c]==0,'y'].mean():.3f})")
print("\n== E. MISSINGNESS PATTERN: is '?' random? ==")
anyq = (df[["workclass","occupation","native_country"]]=="?")
for c in ["workclass","occupation","native_country"]:
    print(f" P(>50K | {c}=='?') = {df.loc[anyq[c],'y'].mean():.3f}  vs known = {df.loc[~anyq[c],'y'].mean():.3f}   n_missing={anyq[c].sum()}")
mq = anyq.any(axis=1)
print(" mean age with ? / without:", df.loc[mq,"age"].mean().round(1), df.loc[~mq,"age"].mean().round(1), "| mean hours:", df.loc[mq,"hours_per_week"].mean().round(1), df.loc[~mq,"hours_per_week"].mean().round(1))
print(" share age>=65 among workclass '?':", (df.loc[df.workclass=="?","age"]>=65).mean().round(3), "vs known:", (df.loc[df.workclass!="?","age"]>=65).mean().round(3), "| share age<=25:", (df.loc[df.workclass=="?","age"]<=25).mean().round(3), "vs", (df.loc[df.workclass!="?","age"]<=25).mean().round(3))
print(" share <=HS among workclass '?':", (df.loc[df.workclass=="?","education_num"]<=9).mean().round(3), "vs known:", (df.loc[df.workclass!="?","education_num"]<=9).mean().round(3))
print(" workclass '?' by sex % female:", (df.loc[df.workclass=="?","sex"]=="Female").mean().round(3), "vs known", (df.loc[df.workclass!="?","sex"]=="Female").mean().round(3))
print("\n== F. RELATIONSHIP WITH TARGET ==")
print("point-biserial r (numeric vs y):"); print(pd.Series({c: stats.pointbiserialr(df.y, df[c])[0] for c in num}).round(3).to_string())
print("\nmean of each numeric by class:"); print(df.groupby("income")[num].mean().round(1).T.to_string())
for c in ["sex","race","marital_status","relationship","workclass","education","occupation"]:
    t=df.groupby(c).y.agg(["mean","size"]).sort_values("mean",ascending=False); t["mean"]=t["mean"].round(3); print(f"\nP(>50K) by {c}:"); print(t.to_string())
df["hours_bin"]=pd.cut(df.hours_per_week,[0,20,34,40,50,60,100]); print("\nP(>50K) by hours bin:"); print(df.groupby("hours_bin",observed=True).y.agg(["mean","size"]).round(3).to_string())
df["age_bin"]=pd.cut(df.age,[16,25,35,45,55,65,91]); print("\nP(>50K) by age bin:"); print(df.groupby("age_bin",observed=True).y.agg(["mean","size"]).round(3).to_string())
print("\n== G. REDUNDANCY / DEPENDENCE ==")
def cv(x,y):
    t=pd.crosstab(x,y); chi=stats.chi2_contingency(t,correction=False)[0]; n=t.values.sum(); k=min(t.shape)-1
    return np.sqrt(chi/(n*k))
cc = ["workclass","education","marital_status","occupation","relationship","race","sex","native_country","income"]
M = pd.DataFrame({a:{b:cv(df[a],df[b]) for b in cc} for a in cc}).round(2); print("Cramer's V:"); print(M.to_string())
print("\nnumeric correlation (Pearson):"); print(df[num].corr().round(2).to_string())
print("Spearman age~education_num:", round(stats.spearmanr(df.age,df.education_num)[0],3), "| capital_gain vs loss (Spearman):", round(stats.spearmanr(df.capital_gain,df.capital_loss)[0],3))
print("\n== H. SAMPLING WEIGHT fnlwgt ==")
print("corr(fnlwgt,y):", round(stats.pointbiserialr(df.y, df.fnlwgt)[0],4), "| fnlwgt by race median:", df.groupby("race").fnlwgt.median().round(0).to_dict())
print("== I. REPRESENTATION / BIAS ==")
print("sex %:", (df.sex.value_counts(normalize=True)*100).round(1).to_dict(), "| race %:", (df.race.value_counts(normalize=True)*100).round(1).to_dict())
print("US native %:", round((df.native_country=="United-States").mean()*100,1), "| native_country levels with <50 rows:", (df.native_country.value_counts()<50).sum())
print("minority cells (income>50K count): Female/Black/Amer-Indian/Other:", {k:int(v) for k,v in df[df.y==1].groupby("race").size().items()})
