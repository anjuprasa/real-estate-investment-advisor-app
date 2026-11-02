import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from model_utils import FEATURES, build_features, train_models

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")
sns.set_theme(style="whitegrid")


@st.cache_data
def load_data():
    return pd.read_csv("clean_data.csv")


@st.cache_resource
def get_models(_df):
    return train_models(_df)


df = load_data()
clf, reg, metrics, importance = get_models(df)

st.title("🏠 Real Estate Investment Advisor")
st.caption("Is a property a good investment, and what will it be worth in 5 years?")

tab_pred, tab_filter, tab_insights, tab_model = st.tabs(
    ["🔮 Predict", "🔎 Filter properties", "📊 Market insights", "🧠 Model details"])

# ------------------------------------------------------------------ Predict
with tab_pred:
    st.subheader("Enter property details")
    with st.form("property_form"):
        c1, c2, c3 = st.columns(3)
        area = c1.number_input("Area (sqft)", 500, 10000, 3000, step=50)
        price = c1.number_input("Current price", 50_000, 3_000_000, 650_000, step=10_000)
        year_built = c1.slider("Year built", 1950, 2026, 2000)
        bedrooms = c2.slider("Bedrooms (BHK)", 1, 8, 3)
        bathrooms = c2.slider("Bathrooms", 1, 6, 2)
        floors = c2.slider("Floors", 1, 5, 2)
        location_score = c3.slider("Location score (0-100)", 0, 100, 70)
        school_score = c3.slider("School score (0-100)", 0, 100, 70)
        submitted = st.form_submit_button("Analyse property", use_container_width=True)

    if submitted:
        row = build_features(area, bedrooms, bathrooms, floors, year_built, location_score, school_score, price)
        proba = clf.predict_proba(row)[0]
        good = proba[1] >= 0.5
        future = float(reg.predict(row)[0])
        rmse = metrics["reg"]["RMSE"]
        growth = (future / price - 1) * 100

        r1, r2, r3 = st.columns(3)
        r1.metric("Investment verdict", "✅ Good Investment" if good else "⚠️ Not a Good Investment")
        r2.metric("Model confidence", f"{max(proba) * 100:.1f}%", help="Probability of the predicted class")
        r3.metric("Estimated price after 5 years", f"{future:,.0f}", f"{growth:+.1f}% vs today")
        st.progress(float(proba[1]), text=f"Probability of being a good investment: {proba[1] * 100:.1f}%")
        st.info(f"Typical error of the price model is about ±{rmse:,.0f} (RMSE), so the likely range is "
                f"{future - rmse:,.0f} to {future + rmse:,.0f}.")

        left, right = st.columns(2)
        with left:
            years = np.arange(0, 6)
            cagr = (future / price) ** (1 / 5)
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.plot(years, price * cagr ** years, marker="o")
            ax.set_xlabel("Years from now"); ax.set_ylabel("Estimated price")
            ax.set_title("Projected price path (constant growth between today and year 5)")
            st.pyplot(fig)
        with right:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            importance.plot.barh(ax=ax, color="steelblue")
            ax.set_title("What drives the verdict (feature importance)")
            st.pyplot(fig)
    else:
        st.info("Fill the form and press **Analyse property**.")

# ------------------------------------------------------------------ Filter
with tab_filter:
    st.subheader("Browse properties in the dataset")
    f1, f2, f3 = st.columns(3)
    a_lo, a_hi = f1.slider("Area (sqft)", int(df.Area_sqft.min()), int(df.Area_sqft.max()),
                           (int(df.Area_sqft.min()), int(df.Area_sqft.max())))
    p_lo, p_hi = f2.slider("Current price", int(df.Current_Price.min()), int(df.Current_Price.max()),
                           (int(df.Current_Price.min()), int(df.Current_Price.max())), step=10_000)
    bhk = f3.multiselect("Bedrooms (BHK)", sorted(df.Bedrooms.unique()), default=sorted(df.Bedrooms.unique()))
    only_good = st.checkbox("Show only properties labelled Good Investment")

    view = df[df.Area_sqft.between(a_lo, a_hi) & df.Current_Price.between(p_lo, p_hi) & df.Bedrooms.isin(bhk)]
    if only_good:
        view = view[view.Good_Investment == 1]
    st.write(f"**{len(view)}** properties match")
    st.dataframe(view[["Property_ID", "Area_sqft", "Bedrooms", "Bathrooms", "Location_Score", "School_Score",
                       "Current_Price", "Price_After_5_Years", "Good_Investment"]], use_container_width=True)

# ------------------------------------------------------------------ Insights
with tab_insights:
    st.subheader("Market insights")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x="Area_sqft", y="Current_Price", hue="Price_Category", alpha=.7, ax=ax)
        ax.set_title("Size vs price"); st.pyplot(fig)
    with c2:
        pivot = df.pivot_table(index="Location_Tier", columns="Bedrooms", values="Price_per_Sqft", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
        ax.set_title("Avg price per sqft: location tier x bedrooms"); st.pyplot(fig)

    c3, c4 = st.columns(2)
    with c3:
        yrs = ["Current_Price", "Price_Year_1", "Price_Year_2", "Price_Year_3", "Price_Year_4", "Price_After_5_Years"]
        st.caption("Average price by category today and in 5 years")
        t = df.groupby("Price_Category")[["Current_Price", "Price_After_5_Years"]].mean().loc[["Low", "Medium", "High"]]
        st.bar_chart(t)
    with c4:
        corr_cols = ["Area_sqft", "Bedrooms", "Property_Age", "Location_Score", "School_Score",
                     "Current_Price", "Infra_Score", "Price_After_5_Years"]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                    annot_kws={"size": 7})
        ax.set_title("Correlation heatmap"); st.pyplot(fig)

# ------------------------------------------------------------------ Model details
with tab_model:
    st.subheader("Held-out test performance (20% split)")
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**Classification: Gradient Boosting**")
        st.dataframe(pd.DataFrame(metrics["clf"], index=["score"]).T.round(3))
    with m2:
        st.markdown("**Regression: Linear Regression (5-year price)**")
        st.dataframe(pd.DataFrame(metrics["reg"], index=["score"]).T.round(3))
    st.caption("The 'Good Investment' label is a multi-factor score (price/sqft vs median, infrastructure score, "
               "bedrooms >= 3, 5-year appreciation vs median). Price growth in this dataset is close to random, so "
               "the appreciation part of the label is hard to predict. Treat results as a decision aid, not advice.")
