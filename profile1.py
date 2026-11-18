import sys, pandas as pd, numpy as np
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 200)
cols = ["age","workclass","fnlwgt","education","education_num","marital_status","occupation",
        "relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country","income"]
path = sys.argv[1]
# keep_default_na=False: do NOT let pandas silently convert anything to NaN; we want to SEE the raw state
df = pd.read_csv(path, header=None, names=cols, keep_default_na=False, na_values=[], skipinitialspace=False)
print("== 1. SHAPE / DTYPES =="); print(df.shape); print(df.dtypes.to_string())
print("\n== 2. TRUE NaN per column (pandas default) =="); print(df.isna().sum().to_string())
cat = df.select_dtypes("object").columns.tolist(); num = [c for c in df.columns if c not in cat]
print("\ncategorical:", cat); print("numeric:", num)
print("\n== 3. HIDDEN MISSING: '?' count per column =="); q = (df[cat]=="?").sum(); print(pd.DataFrame({"n":q,"pct":(q/len(df)*100).round(2)})[q>0].to_string())
print("\nany column with leading/trailing whitespace in values?")
for c in cat: 
    n = (df[c]!=df[c].str.strip()).sum()
    if n: print(" ", c, n)
print("(none printed above = no whitespace problems)")
print("\nrows with >=1 '?':", (df[cat]=="?").any(axis=1).sum(), f"({(df[cat]=='?').any(axis=1).mean()*100:.2f}%)")
print("rows with '?' in workclass AND occupation:", ((df.workclass=="?")&(df.occupation=="?")).sum())
print("workclass '?' but occupation known:", ((df.workclass=="?")&(df.occupation!="?")).sum())
print("occupation '?' but workclass known:", ((df.workclass!="?")&(df.occupation=="?")).sum())
print("\n== 4. TARGET =="); print(df.income.value_counts()); print((df.income.value_counts(normalize=True)*100).round(2))
print("\n== 5. DUPLICATES =="); d = df.duplicated(keep="first").sum(); print("exact duplicate rows (beyond first):", d, f"({d/len(df)*100:.2f}%)")
feat = [c for c in df.columns if c!="income"]
g = df.groupby(feat)["income"].nunique(); print("feature-identical groups with CONFLICTING labels:", (g>1).sum())
print("duplicated fnlwgt values:", df.fnlwgt.duplicated().sum())
print("\n== 6. NUMERIC SUMMARY =="); print(df[num].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T.round(2).to_string())
print("\nskewness:"); print(df[num].skew().round(2).to_string())
print("\nfraction exactly 0:"); print((df[num]==0).mean().round(4).to_string())
for c in ["capital_gain","capital_loss","hours_per_week","age"]:
    print(f"\ntop values of {c}:"); print(df[c].value_counts().head(6).to_string())
print("\ncapital_gain==99999:", (df.capital_gain==99999).sum(), "| income >50K among them:", (df.loc[df.capital_gain==99999,"income"]==">50K").mean().round(4))
print("rows with BOTH gain>0 and loss>0:", ((df.capital_gain>0)&(df.capital_loss>0)).sum())
print("hours_per_week ==99:", (df.hours_per_week==99).sum(), "| <=5:", (df.hours_per_week<=5).sum(), "| >=80:", (df.hours_per_week>=80).sum())
print("age min/max:", df.age.min(), df.age.max(), "| age>=90:", (df.age>=90).sum(), "| age==90:", (df.age==90).sum())
print("\n== 7. CATEGORICAL LEVELS ==")
for c in cat:
    vc = df[c].value_counts(); print(f"\n{c}: {len(vc)} levels; rare(<1%): {(vc/len(df)<0.01).sum()}"); print(vc.head(20).to_string())
