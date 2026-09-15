import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 60)
np.set_printoptions(suppress=True, precision=4)

DATA = r"D:\T-2-1\ML\placement_predict_50k Dataset.csv"   # <-- put the CSV next to this notebook
df = pd.read_csv(DATA)

CAT = ["Gender","City","CollegeTier","Stream","Specialisation",
       "Hostel","HistoryOfBacklogs","CGPA_Tier"]
pd.DataFrame({"column": CAT,
              "n_unique": [df[c].nunique() for c in CAT],
              "categories": [", ".join(map(str, sorted(df[c].unique())))[:70] for c in CAT]})

le = LabelEncoder().fit(df["City"])
print("classes_ (alphabetical):")
for i, cat in enumerate(le.classes_):
    print(f"  {cat:<12} -> {i}")

S = df.loc[:7, ["StudentID","Gender","City"]].copy()
S["Gender_label"] = LabelEncoder().fit_transform(df["Gender"])[:8]
S["City_label"]   = le.transform(S["City"])

# the trap: the encoder just imposed arithmetic on your categories
print("(Ahmedabad + Pune) / 2 =", (0 + 9) / 2, "= 'Hyderabad-ish'   <- meaningless")
print("Pune - Delhi           =", 9 - 3, "'units of city'          <- meaningless")
print("A linear model or a distance metric WILL use these relations.")

oe  = OrdinalEncoder(categories=[["Low","Mid","High"]]).fit(df[["CGPA_Tier"]])
oe2 = OrdinalEncoder(categories=[["Tier3","Tier2","Tier1"]]).fit(df[["CollegeTier"]])
S = df.loc[:7, ["StudentID","CGPA_Tier","CollegeTier"]].copy()
S["CGPA_Tier_ord"]   = oe.transform(S[["CGPA_Tier"]]).astype(int)
S["CollegeTier_ord"] = oe2.transform(S[["CollegeTier"]]).astype(int)

# never assume the order - check it against the target
print(df.groupby("CGPA_Tier", observed=True)["PlacementStatus"].agg(["count","mean"])
        .reindex(["Low","Mid","High"]).round(4))
print()
print(df.groupby("CollegeTier", observed=True)["PlacementStatus"].agg(["count","mean"])
        .reindex(["Tier3","Tier2","Tier1"]).round(4))

ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(df[["City"]])
oh  = ohe.transform(df.loc[:7, ["City"]])

out = pd.DataFrame(oh.astype(int), columns=[c.replace("City_","") for c in ohe.get_feature_names_out()])
out.insert(0, "City", df.loc[:7, "City"].values)
print(out.to_string(index=False))
print("\nrow sums (always 1):", oh.sum(1))

# the dummy-variable trap
ohd = OneHotEncoder(sparse_output=False, drop="first").fit(df[["City"]])
print("drop='first' ->", ohd.transform(df[["City"]]).shape[1], "columns")
print("reference category =", ohe.categories_[0][0], "(now all-zeros)")

# the price in columns and memory
full = OneHotEncoder(sparse_output=False).fit_transform(df[CAT])
print("\nTOTAL one-hot columns from the 8 categorical columns:", full.shape[1])
print(f"dense float64 memory : {full.nbytes/1e6:.1f} MB")
print(f"sparse CSR data      : {OneHotEncoder().fit_transform(df[CAT]).data.nbytes/1e6:.2f} MB")

def binary_encode(series, name):
    cats    = sorted(series.unique())
    mapping = {c: i + 1 for i, c in enumerate(cats)}      # start at 1, like category_encoders
    nbits   = int(np.ceil(np.log2(len(cats) + 1)))
    codes   = series.map(mapping).to_numpy(dtype=int)
    cols    = {f"{name}_{nbits-b}": (codes >> (nbits-1-b)) & 1 for b in range(nbits)}
    return pd.DataFrame(cols), mapping, nbits

bdf, bmap, nbits = binary_encode(df["City"], "City")
print(f"10 cities -> ceil(log2(10+1)) = {nbits} columns (one-hot needed 10)\n")

tbl = pd.DataFrame({"City": list(bmap), "ordinal": list(bmap.values())})
tbl["binary"] = tbl["ordinal"].apply(lambda v: format(v, f"0{nbits}b"))
for b in range(nbits):
    tbl[f"City_{nbits-b}"] = tbl["binary"].str[b].astype(int)

print("\n" + "=" * 70)
print("CATEGORICAL ENCODING IMPACT: M2 LINEAR vs M3 TREE MODELS")
print("=" * 70)

# Compare One-Hot vs Ordinal Encoding on predicting PlacementStatus
y = df["PlacementStatus"]
X_cats = df[CAT].copy().fillna("Missing")

# 1. Label / Ordinal Encoded
X_ord = pd.DataFrame()
for c in CAT:
    X_ord[c] = LabelEncoder().fit_transform(X_cats[c].astype(str))

# 2. One-Hot Encoded
X_ohe = pd.get_dummies(X_cats, drop_first=True)

X_tr_ord, X_te_ord, y_tr, y_te = train_test_split(X_ord, y, test_size=0.2, random_state=42, stratify=y)
X_tr_ohe, X_te_ohe, _, _ = train_test_split(X_ohe, y, test_size=0.2, random_state=42, stratify=y)

enc_results = []

# M2 Logistic Regression
lr_ord = LogisticRegression(max_iter=300).fit(X_tr_ord, y_tr)
lr_ohe = LogisticRegression(max_iter=300).fit(X_tr_ohe, y_tr)

# M3 Random Forest
rf_ord = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42).fit(X_tr_ord, y_tr)
rf_ohe = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42).fit(X_tr_ohe, y_tr)

enc_results.append({
    "Encoding": "Ordinal / Label Encoding",
    "Num Columns": X_ord.shape[1],
    "M2 Logistic Reg Acc": round(accuracy_score(y_te, lr_ord.predict(X_te_ord)), 4),
    "M3 Random Forest Acc": round(accuracy_score(y_te, rf_ord.predict(X_te_ord)), 4)
})

enc_results.append({
    "Encoding": "One-Hot Encoding (drop_first)",
    "Num Columns": X_ohe.shape[1],
    "M2 Logistic Reg Acc": round(accuracy_score(y_te, lr_ohe.predict(X_te_ohe)), 4),
    "M3 Random Forest Acc": round(accuracy_score(y_te, rf_ohe.predict(X_te_ohe)), 4)
})

print(pd.DataFrame(enc_results).to_string(index=False))
print("\nTakeaway: One-Hot encoding avoids false numerical ordering for linear models,")
print("          while tree ensembles can split both representations effectively.")
