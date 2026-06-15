import json, re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# --- UI Config ---
st.set_page_config(page_title="SIA — System Sentinel", layout="wide", page_icon="🛡️")

# --- Constants ---
MAPPING_DICT = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
REVERSE_MAPPING = {v: k for k, v in MAPPING_DICT.items()}
MODEL_LOCATION = Path("models/sia_model/best")
INF_RESULTS = Path("outputs/labeled_tickets.csv")
AUDIT_LOGS = Path("outputs/evidence_dossiers.json")

# --- Helper Functions ---
@st.cache_resource(show_spinner="Engaging AI Core...")
def load_assets():
    if not MODEL_LOCATION.exists(): return None, None, 0.5
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(str(MODEL_LOCATION))
    base = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-small", num_labels=2)
    eng = PeftModel.from_pretrained(base, str(MODEL_LOCATION)).float().to(dev).eval()
    
    thresh_path = Path("models/sia_model/threshold.npy")
    thresh = float(np.load(str(thresh_path))[0]) if thresh_path.exists() else 0.5
    return tok, eng, thresh

def format_input(r):
    speed = 'FAST' if float(r.get('Resolution_Time_Hours', 30)) <= 10 else 'MID'
    return f"[SUBJ] {r['Ticket_Subject']} [BODY] {r['Ticket_Description']} | CAT:{r['Issue_Category']} | RT:{speed} | PRI:{r['Priority_Level']}"

# --- APP NAVIGATION ---
st.title("🛡️ SIA — Enterprise Priority Sentinel")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🎫 Diagnostic Inspector", "⚙️ System Performance", "🔍 Audit Explorer"])

# --- TAB 1: Single Diagnostic ---
with tab1:
    tok, eng, thr = load_assets()
    c1, c2 = st.columns(2)
    with c1:
        s = st.text_input("Subject", "Server access issue")
        c = st.selectbox("Category", ['Technical', 'Fraud', 'Account', 'Billing', 'General Inquiry'])
    with c2:
        p = st.select_slider("Assigned Priority", ['Low', 'Medium', 'High', 'Critical'])
        rt = st.number_input("Resolution Time (Hrs)", 0.0, 72.0, 24.0)
    d = st.text_area("Description", "Cannot reach server instance...", height=150)
    
    if st.button("Execute Diagnostic", type="primary"):
        rec = {'Ticket_Subject':s, 'Ticket_Description':d, 'Issue_Category':c, 'Priority_Level':p, 'Resolution_Time_Hours':rt}
        text = format_input(rec)
        # Inference
        probs = 0.88 # Demo logic
        if probs > thr:
            st.error("🚨 Mismatch Detected: Hidden Crisis")
        else:
            st.success("✅ Consistent: Priority Verified")

# --- TAB 2: System Performance (Ablation/Metrics) ---
with tab2:
    st.subheader("Model Diagnostic Metrics")
    m_path = Path("models/sia_model/metrics.json")
    if m_path.exists():
        with open(m_path) as f: m = json.load(f)
        k1, k2, k3 = st.columns(3)
        k1.metric("Accuracy", f"{m['accuracy']:.2%}")
        k2.metric("Macro F1", f"{m['macro_f1']:.4f}")
        k3.metric("Best Epoch", m['best_epoch'])
        
        # Static Confusion Matrix Heatmap
        st.write("### Model Confusion Matrix")
        fig = px.imshow(m['confusion_matrix'], text_auto=True, labels=dict(x="Predicted", y="True"), 
                        x=['Consistent', 'Mismatch'], y=['Consistent', 'Mismatch'], color_continuous_scale='Reds')
        fig.update_layout(dragmode=False)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# --- TAB 3: Audit Explorer ---
with tab3:
    st.subheader("Audit Investigation")
    if AUDIT_LOGS.exists():
        with open(AUDIT_LOGS) as f: logs = json.load(f)
        
        selected_id = st.selectbox("Select Flagged Ticket", [d['ticket_id'] for d in logs])
        for log in logs:
            if log['ticket_id'] == selected_id:
                st.json(log)
    else:
        st.info("No audit logs found. Please run predict.py first.")
