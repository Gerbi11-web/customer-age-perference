import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

st.set_page_config(page_title="Customer Age Preferences", page_icon="📊", layout="wide")
st.title("📊 Customer Age Preferences")
st.caption("Customer intelligence • Product performance • Machine-learning insights")

with st.sidebar:
    st.header("About")
    st.write("Upload a clothing dataset to explore age groups, product/category metrics, and recommendation prediction.")
    st.info("Review activity is not sales. Sales/revenue insights appear only if those fields exist in your data.")

file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

def read_file(f):
    return pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)

def find_col(df, aliases):
    cols = {str(c).strip().lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in cols:
            return cols[a.lower()]
    for normalized, original in cols.items():
        if any(a.lower() in normalized for a in aliases):
            return original
    return None

def numeric(s):
    return pd.to_numeric(s, errors="coerce")

if not file:
    st.info("Upload your dataset above to begin.")
    st.markdown("**Useful columns:** `Age`, `Class Name` or `Category`, `Rating`, `Recommended IND`. For actual business performance, include `Units Sold`, `Revenue`, and optionally `Profit`.")
    st.stop()

try:
    raw = read_file(file)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

if raw.empty:
    st.error("The uploaded file contains no rows.")
    st.stop()

st.subheader("1. Dataset overview")
a, b, c = st.columns(3)
a.metric("Rows", f"{len(raw):,}")
b.metric("Columns", len(raw.columns))
c.metric("Duplicate rows", f"{raw.duplicated().sum():,}")
st.dataframe(raw.head(10), use_container_width=True)
st.caption("Columns found: " + ", ".join(map(str, raw.columns)))

age = find_col(raw, ["Age", "Customer Age", "Age of Customer"])
category = find_col(raw, ["Class Name", "Category", "Product Category", "Clothing Category", "Department Name", "Product"])
rating = find_col(raw, ["Rating", "Review Rating", "Product Rating"])
target = find_col(raw, ["Recommended IND", "Recommended", "Recommend", "Would Recommend"])
units = find_col(raw, ["Units Sold", "Quantity Sold", "Quantity", "Units", "Sales Quantity"])
revenue = find_col(raw, ["Revenue", "Sales Revenue", "Total Revenue", "Sales Amount", "Amount"])
profit = find_col(raw, ["Profit", "Gross Profit", "Net Profit"])

st.subheader("2. Detected metrics")
detected = [("Customer age", age), ("Product/category", category), ("Rating", rating),
            ("Recommendation label", target), ("Units sold", units), ("Revenue", revenue), ("Profit", profit)]
cols = st.columns(4)
for i, (name, col) in enumerate(detected):
    cols[i % 4].metric(name, "Found" if col else "Not found")

data = raw.copy()
if age:
    data[age] = numeric(data[age])
    bins = [0, 17, 24, 34, 44, 54, 64, 120]
    labels = ["Under 18", "18–24", "25–34", "35–44", "45–54", "55–64", "65+"]
    data["Age Group"] = pd.cut(data[age], bins=bins, labels=labels, include_lowest=True)

st.subheader("3. Customer intelligence")
if age:
    age_counts = data["Age Group"].value_counts().reindex(labels, fill_value=0)
    if age_counts.sum():
        left, right = st.columns([1, 2])
        left.metric("Largest age group", str(age_counts.idxmax()))
        left.metric("Records in group", f"{int(age_counts.max()):,}")
        fig, ax = plt.subplots()
        age_counts.plot(kind="bar", ax=ax)
        ax.set_title("Records by age group")
        ax.set_xlabel("Age group")
        ax.set_ylabel("Record count")
        ax.tick_params(axis="x", rotation=35)
        right.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("No usable age values found.")
else:
    st.info("Add an Age column to enable age analysis.")

st.subheader("4. Product/category performance")
if category:
    work = data.dropna(subset=[category]).copy()
    agg = {"Record count": (category, "size")}
    if units:
        work[units] = numeric(work[units])
        agg["Units sold"] = (units, "sum")
    if revenue:
        work[revenue] = numeric(work[revenue])
        agg["Revenue"] = (revenue, "sum")
    if profit:
        work[profit] = numeric(work[profit])
        agg["Profit"] = (profit, "sum")
    if rating:
        work[rating] = numeric(work[rating])
        agg["Average rating"] = (rating, "mean")
    if target:
        t = work[target]
        if t.dtype == object:
            work["_target_num"] = t.astype(str).str.strip().str.lower().map({
                "yes": 1, "true": 1, "1": 1, "recommended": 1,
                "no": 0, "false": 0, "0": 0, "not recommended": 0
            })
        else:
            work["_target_num"] = numeric(t)
        agg["Recommendation rate (%)"] = ("_target_num", "mean")
    table = work.groupby(category, dropna=True).agg(**agg).reset_index()
    if "Recommendation rate (%)" in table:
        table["Recommendation rate (%)"] *= 100
    sortby = next((x for x in ["Revenue", "Units sold", "Record count"] if x in table), None)
    if sortby:
        table = table.sort_values(sortby, ascending=False)
    st.dataframe(table.round(2), use_container_width=True)
    chart_metric = next((x for x in ["Revenue", "Units sold", "Record count"] if x in table), None)
    if chart_metric and not table.empty:
        top = table.head(12).sort_values(chart_metric)
        fig, ax = plt.subplots()
        ax.barh(top[category].astype(str), top[chart_metric])
        ax.set_title(f"Top categories by {chart_metric.lower()}")
        ax.set_xlabel(chart_metric)
        st.pyplot(fig)
        plt.close(fig)
else:
    st.info("Add a category column such as Class Name or Category.")

st.subheader("5. Age group × category")
if age and category:
    cross_data = data.dropna(subset=["Age Group", category])
    if not cross_data.empty:
        cross = pd.crosstab(cross_data["Age Group"], cross_data[category])
        st.dataframe(cross, use_container_width=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        cross.plot(kind="bar", stacked=True, ax=ax, legend=False)
        ax.set_title("Records by age group and category")
        ax.set_xlabel("Age group")
        ax.set_ylabel("Record count")
        ax.tick_params(axis="x", rotation=35)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("These are dataset record counts, not confirmed sales unless each row represents a transaction.")
else:
    st.info("Age × category analysis requires both age and category columns.")

st.subheader("6. Machine-learning evaluation")
st.write("This classifier predicts a recommendation label. It does not predict future sales.")
if age and category and rating and target:
    ml = data[[age, category, rating, target]].copy()
    ml[age] = numeric(ml[age])
    ml[rating] = numeric(ml[rating])
    if ml[target].dtype == object:
        ml[target] = ml[target].astype(str).str.strip().str.lower().map({
            "yes": 1, "true": 1, "1": 1, "recommended": 1,
            "no": 0, "false": 0, "0": 0, "not recommended": 0
        })
    ml[target] = numeric(ml[target])
    ml = ml.dropna()
    ml = ml[ml[target].isin([0, 1])]
    ml[target] = ml[target].astype(int)
    if len(ml) >= 50 and ml[target].nunique() == 2 and ml[target].value_counts().min() >= 2:
        X, y = ml[[age, category, rating]], ml[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        prep = ColumnTransformer([
            ("numeric", "passthrough", [age, rating]),
            ("category", OneHotEncoder(handle_unknown="ignore"), [category])
        ])
        model = Pipeline([("prep", prep), ("classifier", RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ))])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
        base_pred = baseline.predict(X_test)
        m = st.columns(4)
        m[0].metric("Accuracy", f"{accuracy_score(y_test, pred):.3f}")
        m[1].metric("Precision", f"{precision_score(y_test, pred, zero_division=0):.3f}")
        m[2].metric("Recall", f"{recall_score(y_test, pred, zero_division=0):.3f}")
        m[3].metric("F1-score", f"{f1_score(y_test, pred, zero_division=0):.3f}")
        st.write("Baseline comparison")
        st.dataframe(pd.DataFrame({
            "Metric": ["Accuracy", "F1-score"],
            "Baseline": [accuracy_score(y_test, base_pred), f1_score(y_test, base_pred, zero_division=0)],
            "Random Forest": [accuracy_score(y_test, pred), f1_score(y_test, pred, zero_division=0)]
        }).round(3), use_container_width=True)
        with st.expander("Classification report and confusion matrix"):
            st.dataframe(pd.DataFrame(classification_report(y_test, pred, output_dict=True, zero_division=0)).transpose().round(3))
            cm = confusion_matrix(y_test, pred)
            st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))
        st.caption("Rating is closely related to recommendation and may inflate usefulness for some real-world tasks. Compare against a model without rating before relying on predictions.")
    else:
        st.warning("Not enough usable labelled rows or both target classes are not present.")
else:
    st.info("ML evaluation requires Age, a category column, Rating, and a binary recommendation label (such as Recommended IND).")

st.subheader("7. Export")
st.download_button("Download data as CSV", data=data.to_csv(index=False).encode("utf-8"),
                   file_name="customer_age_preferences_data.csv", mime="text/csv")
