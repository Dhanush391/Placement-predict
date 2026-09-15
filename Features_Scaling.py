import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   MaxAbsScaler, Normalizer, PowerTransformer, QuantileTransformer)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import clone

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 50)
np.set_printoptions(suppress=True, precision=4)

DATA = r"D:\T-2-1\ML\placement_predict_50k Dataset.csv"

df = pd.read_csv(DATA)
print(df.shape)
df.head()

NUM = ["CGPA", "AttendancePercent", "AptitudeTestScore", "CodingTestScore", "Salary Package"]
df[NUM].agg(["min", "max", "mean", "std"]).T.round(3)

# Euclidean distance between student 1 and student 2 on two raw features
a = df.loc[0, ["CGPA", "AttendancePercent"]].values.astype(float)
b = df.loc[1, ["CGPA", "AttendancePercent"]].values.astype(float)
contrib = (a - b) ** 2
print("A =", a, "  B =", b)
print("(CGPA diff)^2   =", round(contrib[0], 4))
print("(Attend diff)^2 =", round(contrib[1], 4))
print("euclidean       =", round(np.sqrt(contrib.sum()), 4))
print(f"attendance share of the distance = {100*contrib[1]/contrib.sum():.2f} %")

cg = df["CGPA"]
mu, sd_pop, sd_samp = cg.mean(), cg.std(ddof=0), cg.std(ddof=1)
print(f"n     = {len(cg)}")
print(f"mu    = {mu:.6f}")
print(f"sigma = {sd_pop:.6f}   (ddof=0, what sklearn uses)")
print(f"sigma = {sd_samp:.6f}   (ddof=1, what pandas uses)")

# hand calculation for the first five students
hand = df.loc[:4, ["StudentID", "CGPA"]].copy()
hand["z = (x-mu)/sigma"] = ((hand["CGPA"] - mu) / sd_pop).round(4)


scaler = StandardScaler()
scaler.fit(df[["CGPA"]])              # LEARN mu and sigma
z = scaler.transform(df[["CGPA"]])    # APPLY the formula
print("mean_ :", scaler.mean_)
print("scale_:", scaler.scale_)
print("var_  :", scaler.var_)
print("first 5:", z[:5].ravel())
print(f"after -> mean = {z.mean():.10f}   std = {z.std():.10f}")
print(f"range -> min  = {z.min():.4f}   max = {z.max():.4f}")

# all five columns at once
X = df[NUM].fillna(df[NUM].median())
Z = StandardScaler().fit_transform(X)
print(pd.DataFrame(Z[:5], columns=NUM).round(4).to_string(index=False))
print("\nmeans after:", np.round(Z.mean(0), 12))
print("stds  after:", np.round(Z.std(0), 6))

xmin, xmax = cg.min(), cg.max()
print(f"min = {xmin}   max = {xmax}   range = {xmax - xmin}")

hand = df.loc[:4, ["StudentID", "CGPA"]].copy()
hand["x_scaled"] = ((hand["CGPA"] - xmin) / (xmax - xmin)).round(4)
print(hand.to_string(index=False))

mm = MinMaxScaler().fit(df[["CGPA"]])
m = mm.transform(df[["CGPA"]])
print("data_min_ :", mm.data_min_, "  data_max_ :", mm.data_max_)
print("scale_    :", mm.scale_,    "  min_      :", mm.min_)
print("first 5   :", m[:5].ravel())
print(f"min = {m.min():.4f}  max = {m.max():.4f}  mean = {m.mean():.4f}")

m2 = MinMaxScaler(feature_range=(-1, 1)).fit_transform(df[["CGPA"]])
print("feature_range=(-1,1) first 5:", m2[:5].ravel())

s  = df["Salary Package"].copy()
s2 = s.copy(); s2.iloc[7] = 2600.0          # the typo
X2 = s2.values.reshape(-1, 1)
ok = np.ones(len(s), bool); ok[7] = False   # everyone except the corrupted row

rows = []
for name, Sc in [("MinMaxScaler", MinMaxScaler), ("StandardScaler", StandardScaler), ("RobustScaler", RobustScaler)]:
    clean = Sc().fit_transform(s.values.reshape(-1, 1)).ravel()
    dirty = Sc().fit_transform(X2).ravel()
    rows.append([name,
                 round(clean[ok].max() - clean[ok].min(), 4),
                 round(dirty[ok].max() - dirty[ok].min(), 4)])
t = pd.DataFrame(rows, columns=["scaler", "clean width", "width after the typo"])
t["% of range kept"] = (100 * t["width after the typo"] / t["clean width"]).round(1)

X = df[["CodingTestScore"]].fillna(df["CodingTestScore"].median())
q1, q2, q3 = np.percentile(X.values, [25, 50, 75])
print(f"Q1 = {q1}   median = {q2}   Q3 = {q3}   IQR = {q3-q1:.4f}")

hand = df.loc[:4, ["StudentID", "CodingTestScore"]].copy()
hand["robust"] = ((hand["CodingTestScore"] - q2) / (q3 - q1)).round(4)
print(hand.to_string(index=False))

rb = RobustScaler().fit(X)
r = rb.transform(X)
print("center_ (median):", rb.center_)
print("scale_  (IQR)   :", rb.scale_)
print("first 5         :", r[:5].ravel())
print(f"median after = {np.median(r):.6f}   IQR after = {np.percentile(r,75)-np.percentile(r,25):.6f}")

# head-to-head on the same column
cmp = pd.DataFrame({
    "raw":      X.values.ravel(),
    "standard": StandardScaler().fit_transform(X).ravel(),
    "minmax":   MinMaxScaler().fit_transform(X).ravel(),
    "robust":   RobustScaler().fit_transform(X).ravel(),
    "maxabs":   MaxAbsScaler().fit_transform(X).ravel()})
cmp.describe().T.round(4)

ma = MaxAbsScaler().fit(df[["Salary Package"]])
print("max_abs_:", ma.max_abs_)
print("raw    :", df["Salary Package"].head().values)
print("scaled :", ma.transform(df[["Salary Package"]])[:5].ravel())

print("\n" + "=" * 70)
print("M2 vs M3: WHEN DOES FEATURE SCALING MATTER?")
print("=" * 70)
print("Hypothesis: M2 Linear/Gradient Descent models are scale-sensitive.")
print("            M3 Decision Trees are monotonic and scale-invariant.\n")

# Test on numerical subset
num_cols = ["CGPA", "AttendancePercent", "AptitudeTestScore", "CodingTestScore"]
X_raw = df[num_cols].fillna(df[num_cols].median())
y = df["PlacementStatus"]

X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)

scalers = {
    "Raw (Unscaled)": None,
    "StandardScaler (Z-Score)": StandardScaler(),
    "MinMaxScaler ([0, 1])": MinMaxScaler()
}

results = []
for s_name, sc in scalers.items():
    if sc:
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
    else:
        X_tr_s, X_te_s = X_tr.values, X_te.values

    # M2 Linear Model
    lr = LogisticRegression(max_iter=200, random_state=42)
    lr.fit(X_tr_s, y_tr)
    lr_acc = round(accuracy_score(y_te, lr.predict(X_te_s)), 4)

    # M3 Tree Model
    dt = DecisionTreeClassifier(max_depth=6, random_state=42)
    dt.fit(X_tr_s, y_tr)
    dt_acc = round(accuracy_score(y_te, dt.predict(X_te_s)), 4)

    results.append({
        "Scaler": s_name,
        "M2 Logistic Reg Accuracy": lr_acc,
        "M3 Decision Tree Accuracy": dt_acc,
        "Tree Changed?": "NO (Invariant)" if len(results) > 0 and dt_acc == results[0]["M3 Decision Tree Accuracy"] else "Base"
    })

print(pd.DataFrame(results).to_string(index=False))
print("\nTakeaway: Scaling drastically stabilizes linear models and gradient descent,")
print("          while decision trees make identical splits regardless of scaling.")