import pandas as pd
import numpy as np
import joblib
import streamlit as st
import shap

st.set_page_config(page_title="Member Retention Advisor", page_icon="🏥", layout="wide")

pipelines = joblib.load("models/final_pipelines.pkl")
weights = joblib.load("models/final_weights.pkl")
input_cols = joblib.load("models/final_input_columns.pkl")

df = pd.read_csv("data/final_health_churn.csv")
preds = pd.read_csv("data/all_predictions.csv")

MEMBER_VALUE_YEAR = 1800.0

def risk_label(p):
    if p >= 0.7: return "HIGH"
    if p >= 0.4: return "MEDIUM"
    return "LOW"

def map_action(feature):
    f = feature.lower()
    if "days_since_last_visit" in f or "satisfaction" in f or "missed_appointments" in f:
        return "Care Outreach", "Re-engage member via nurse line / care coordinator"
    if "cost" in f or "premium" in f or "billing" in f or "denial" in f or "prior_auth" in f:
        return "Benefit Education", "Explain coverage, savings programs, appeal denied claims"
    if "pharmacy" in f or "adherence" in f or "medication" in f:
        return "Pharmacy Support", "Medication adherence program / mail-order enrollment"
    if "grievance" in f or "service" in f or "star_rating" in f or "rural" in f:
        return "Service Recovery", "Resolve complaints, improve access, escalate to retention team"
    if "plan" in f:
        return "Benefit Education", "Educate on plan benefits and alternatives"
    return "Care Outreach", "Standard retention touchpoint"

@st.cache_data
def precompute_shap(version):
    pipe_xgb = pipelines["XGBoost"]
    x_all = preds[input_cols]
    x_all_t = pipe_xgb.named_steps["pre"].transform(x_all)
    explainer = shap.TreeExplainer(pipe_xgb.named_steps["model"])
    sv = explainer.shap_values(x_all_t)
    fnames = pipe_xgb.named_steps["pre"].get_feature_names_out()
    top_idx = np.argsort(-sv, axis=1)
    top1 = fnames[top_idx[:, 0]]
    top2 = fnames[top_idx[:, 1]]
    top3 = fnames[top_idx[:, 2]]
    return top1, top2, top3

top1, top2, top3 = precompute_shap("v3")
preds["Top_Driver"] = top1
preds["Driver_2"] = top2
preds["Driver_3"] = top3

preds["Risk"] = preds["Churn_Probability"].apply(risk_label)
n_high = int((preds["Risk"] == "HIGH").sum())
n_med = int((preds["Risk"] == "MEDIUM").sum())
n_low = int((preds["Risk"] == "LOW").sum())

st.title("🏥 Health Plan Member Retention Advisor")
st.caption("Predict → Explain → Act")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Overview", "🩺 Member Risk List", "🎯 Retention Advisor",
     "🧠 Member Detail", "💰 Business Impact", "🧪 Model Comparison"])

with tab1:
    st.subheader("How many members are at risk of leaving?")
    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 High risk — likely to leave", f"{n_high:,} members", f"{n_high/len(preds)*100:.1f}% of members")
    c2.metric("🟡 Medium risk — watch", f"{n_med:,} members", f"{n_med/len(preds)*100:.1f}% of members")
    c3.metric("🟢 Low risk — safe", f"{n_low:,} members", f"{n_low/len(preds)*100:.1f}% of members")

    st.subheader("Risk distribution")
    buckets = pd.cut(preds["Churn_Probability"], bins=[0, 0.4, 0.7, 1.0],
                     labels=["Low (0-40%)", "Medium (40-70%)", "High (70-100%)"])
    counts = buckets.value_counts().reindex(["Low (0-40%)", "Medium (40-70%)", "High (70-100%)"])
    st.bar_chart(counts)

    st.info(f"**Action needed:** {n_high + n_med:,} members ({((n_high+n_med)/len(preds))*100:.1f}%) show churn risk "
            f"— see the **Retention Advisor** tab for recommended actions per member.")

with tab2:
    risk_filter = st.radio("Filter risk level", ["ALL", "HIGH", "MEDIUM", "LOW"], horizontal=True)
    view = preds.copy()
    if risk_filter != "ALL":
        view = view[view["Risk"] == risk_filter]
    view = view.sort_values("Churn_Probability", ascending=False)
    show = view[["MemberID", "Age", "Plan_Type", "City", "Churn_Probability", "Risk"]].copy()
    show["Churn_Probability"] = show["Churn_Probability"].apply(lambda p: f"{p*100:.1f}%")
    st.dataframe(show.head(300), hide_index=True, use_container_width=True)
    st.info(f"{len(view):,} members in this view — sorted by risk, highest first")

with tab3:
    st.subheader("Retention Advisor — recommended actions")
    actions = preds["Top_Driver"].apply(lambda f: map_action(f)[0])
    counts = actions.value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Care Outreach", f"{counts.get('Care Outreach', 0):,} members")
    c2.metric("Benefit Education", f"{counts.get('Benefit Education', 0):,} members")
    c3.metric("Pharmacy Support", f"{counts.get('Pharmacy Support', 0):,} members")
    c4.metric("Service Recovery", f"{counts.get('Service Recovery', 0):,} members")
    st.markdown("---")
    st.subheader("High-risk members and their assigned action")
    high = preds[preds["Risk"] == "HIGH"].sort_values("Churn_Probability", ascending=False).head(200)
    rows = []
    for _, row in high.iterrows():
        action, detail = map_action(row["Top_Driver"])
        rows.append({"MemberID": row["MemberID"],
                     "Churn Risk": f"{row['Churn_Probability']*100:.1f}%",
                     "Key Driver": row["Top_Driver"].replace("num__", "").replace("cat__", ""),
                     "Recommended Action": action,
                     "What to do": detail})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

with tab4:
    member_id = st.selectbox("Select a member", preds.sort_values("Churn_Probability", ascending=False)["MemberID"])
    row = preds[preds["MemberID"] == member_id].iloc[0]
    proba = row["Churn_Probability"]
    risk = risk_label(proba)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Churn probability", f"{proba*100:.1f}%", delta=f"{risk} RISK")
        st.write(f"Age: {row['Age']}  |  Sex: {row['Sex']}  |  Plan: {row['Plan_Type']}  |  City: {row['City']}")
    with col2:
        if risk == "HIGH":
            st.error("Immediate retention action recommended")
        elif risk == "MEDIUM":
            st.warning("Monitor and engage")
        else:
            st.success("Low risk — no action needed")
    st.subheader("Why? (plain-language drivers)")
    for feat in [row["Top_Driver"], row["Driver_2"], row["Driver_3"]]:
        st.markdown(f"- **{feat.replace('num__', '').replace('cat__', '')}**")
    st.subheader("Recommended retention action")
    action, detail = map_action(row["Top_Driver"])
    st.markdown(f"### ✅ {action}")
    st.write(detail)

with tab5:
    st.subheader("What happens if we act on high-risk members?")
    high = preds[preds["Churn_Probability"] >= 0.7]
    saved_frac = st.slider("Estimated retention success rate (%)", 5, 60, 30)
    saved = len(high) * saved_frac / 100
    revenue = saved * MEMBER_VALUE_YEAR
    c1, c2, c3 = st.columns(3)
    c1.metric("High-risk members flagged", f"{len(high):,}")
    c2.metric(f"Saved at {saved_frac}% success", f"{saved:.0f} members")
    c3.metric("Revenue preserved (yearly)", f"${revenue:,.0f}")
    st.caption(f"Assumes average member value of ${MEMBER_VALUE_YEAR:,.0f}/year. Outreach costs not subtracted — conservative.")

with tab6:
    st.subheader("How the final model was built")
    st.write("We tested 8 algorithms on the same test set. 4 passed the quality bar and were combined "
             "into a weighted ensemble (weights = cross-validated AUC).")
    st.dataframe(pd.DataFrame([
        {"Algorithm": "Logistic Regression", "Accuracy": "83.9%", "ROC-AUC": "83.6%", "In final model": "Yes"},
        {"Algorithm": "Gradient Boosting", "Accuracy": "83.3%", "ROC-AUC": "83.6%", "In final model": "Yes"},
        {"Algorithm": "Random Forest", "Accuracy": "82.3%", "ROC-AUC": "82.5%", "In final model": "Yes"},
        {"Algorithm": "XGBoost", "Accuracy": "82.3%", "ROC-AUC": "80.4%", "In final model": "Yes"},
        {"Algorithm": "Naive Bayes / KNN / SVM / Decision Tree", "Accuracy": "—", "ROC-AUC": "< 80%", "In final model": "No (tested)"},
        {"Algorithm": "Weighted Ensemble (FINAL)", "Accuracy": "83.6%", "ROC-AUC": "83.7%", "In final model": "Yes"},
    ]), hide_index=True, use_container_width=True)

st.caption("Prototype for Cognizant NPN Hackathon — Member Churn Prediction & Retention Advisor")