import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
from pathlib import Path

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Vendor Invoice Intelligence", page_icon="📊", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "inventory.db"
MODEL_DIR = BASE_DIR / "models"
FREIGHT_MODEL_PATH = MODEL_DIR / "predict_freight_model.pkl"
RISK_MODEL_PATH = MODEL_DIR / "predict_flag_invoice_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# ---------------- HELPERS ----------------
@st.cache_data
def load_tables(db_path):
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path))
    try:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )["name"].tolist()
        data = {}
        for table in tables:
            # Table names come from SQLite metadata, not user input.
            data[table] = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
        return data
    finally:
            conn.close()

def load_model(path):
    try:
        return joblib.load(path) if path.exists() else None
    except Exception as e:
        st.error(f"Could not load {path.name}: {e}")
        return None

def risk_message(flag):
    if int(flag) == 1:
        st.warning("⚠️ Manual approval recommended — this invoice is flagged as potentially risky.")
    else:
        st.success("✅ Invoice looks normal based on the model.")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Vendor Invoice Intelligence")
page = st.sidebar.radio("Navigation", ["Dashboard", "Freight Prediction", "Invoice Risk Flagging", "Database Explorer", "About"])

tables = load_tables(DB_PATH)
vendor_df = tables.get("vendor_invoice") if tables else None

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.title("📊 Vendor Invoice Intelligence System")
    st.caption("Freight Cost Prediction & Invoice Risk Flagging")

    if vendor_df is None:
        st.info("Place inventory.db in the same folder as app.py to view database-powered dashboard data.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Vendor invoices", f"{len(vendor_df):,}")
        c2.metric("Total invoice value", f"{pd.to_numeric(vendor_df.get('Dollars'), errors='coerce').sum():,.2f}" if "Dollars" in vendor_df else "N/A")
        c3.metric("Total freight", f"{pd.to_numeric(vendor_df.get('Freight'), errors='coerce').sum():,.2f}" if "Freight" in vendor_df else "N/A")

        st.subheader("Invoice data preview")
        st.dataframe(vendor_df.head(20), use_container_width=True)

        if {"Dollars", "Freight"}.issubset(vendor_df.columns):
            plot_df = vendor_df[["Dollars", "Freight"]].apply(pd.to_numeric, errors="coerce").dropna()
            st.subheader("Invoice value vs freight")
            st.scatter_chart(plot_df, x="Dollars", y="Freight")

        if {"Quantity", "Freight"}.issubset(vendor_df.columns):
            plot_df = vendor_df[["Quantity", "Freight"]].apply(pd.to_numeric, errors="coerce").dropna()
            st.subheader("Quantity vs freight")
            st.scatter_chart(plot_df, x="Quantity", y="Freight")

# ---------------- FREIGHT PREDICTION ----------------
elif page == "Freight Prediction":
    st.title("🚚 Freight Cost Prediction")
    st.write("Predict freight cost from the invoice dollar value, using the trained regression model.")

    model = load_model(FREIGHT_MODEL_PATH)
    if model is None:
        st.error("Freight model not found. Train your model in the notebook and save it to models/predict_freight_model.pkl.")
    else:
        dollars = st.number_input("Invoice dollar value", min_value=0.0, value=18500.0, step=100.0)
        if st.button("Predict Freight", type="primary"):
            try:
                prediction = model.predict(pd.DataFrame({"Dollars": [dollars]}))[0]
                st.metric("Predicted freight", f"{prediction:,.2f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---------------- RISK FLAGGING ----------------
elif page == "Invoice Risk Flagging":
    st.title("🛡️ Invoice Risk Flagging")
    st.write("Estimate whether an invoice should be flagged for manual approval.")

    model = load_model(RISK_MODEL_PATH)
    scaler = load_model(SCALER_PATH)
    if model is None or scaler is None:
        st.error("Risk model and/or scaler missing. Train and save both in the models folder.")
    else:
        invoice_quantity = st.number_input("Invoice quantity", min_value=0.0, value=50.0)
        invoice_dollars = st.number_input("Invoice dollars", min_value=0.0, value=352.95, step=1.0)
        freight = st.number_input("Freight", min_value=0.0, value=1.73, step=0.1)
        total_item_quantity = st.number_input("Total item quantity (from purchases)", min_value=0.0, value=162.0)
        total_item_dollars = st.number_input("Total item dollars (from purchases)", min_value=0.0, value=2476.0, step=1.0)

        if st.button("Check Invoice Risk", type="primary"):
            try:
                features = pd.DataFrame([{
                    "invoice_quantity": invoice_quantity,
                    "invoice_dollars": invoice_dollars,
                    "Freight": freight,
                    "total_item_quantity": total_item_quantity,
                    "total_item_dollars": total_item_dollars,
                }])
                scaled = scaler.transform(features)
                prediction = int(model.predict(scaled)[0])
                risk_message(prediction)
                st.metric("Model classification", "Flagged (1)" if prediction else "Normal (0)")
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(scaled)[0]
                    st.write(f"Flag probability: **{probabilities[prediction]:.1%}** (probability for predicted class)")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---------------- DATABASE EXPLORER ----------------
elif page == "Database Explorer":
    st.title("🗃️ Database Explorer")
    if not tables:
        st.error("inventory.db not found beside app.py.")
    else:
        selected_table = st.selectbox("Choose a table", list(tables.keys()))
        data = tables[selected_table]
        st.caption(f"{len(data):,} rows · {len(data.columns)} columns")
        st.dataframe(data, use_container_width=True)
        st.download_button(
            "Download table as CSV",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_table}.csv",
            mime="text/csv"
        )

# ---------------- ABOUT ----------------
elif page == "About":
    st.title("ℹ️ About this project")
    st.title("ℹ️ About this project")
    st.markdown(
        "**Vendor Invoice Intelligence System** has two machine-learning components:\n\n"
        "- **Freight Cost Prediction (Regression):** estimates freight charges from invoice dollar value.\n"
        "- **Invoice Risk Flagging (Classification):** predicts whether an invoice may require manual review using invoice and purchase-summary features.\n\n"
        "The risk label in the original notebook is rule-derived (invoice-to-purchase dollar mismatch or receiving delay above 10 days).\n"
        "The classifier therefore learns that rule-based label rather than independently verified fraud or risk."
    )
    st.info("This app is a decision-support tool. Review flagged invoices manually; a model prediction is not proof of an error or fraud.")
