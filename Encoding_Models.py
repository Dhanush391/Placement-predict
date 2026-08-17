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
