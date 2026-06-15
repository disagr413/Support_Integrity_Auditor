import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Set page config at the very top
st.set_page_config(page_title="SIA — Support Integrity Auditor", layout="wide", page_icon="🔍")

# --- Constants ---
PMAP  = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
RPMAP = {v: k for k, v in PMAP.items()}
BASE  = "microsoft/deberta-v3-small"
MDL   = Path("models/sia_model")

ESCA_EV = [
    (r'\bfraud\w*\b', 'fraud_indicator', 0.35), (r'\bphish\w*\b', 'security_threat', 0.35),
    (r'\bhack\w*\b', 'security_threat', 0.30), (r'\bstolen\b', 'security_threat', 0.30),
    (r'\bunauthori[sz]ed\b', 'security_threat', 0.28), (r'\bdata\s+breach\b', 'data_risk', 0.32),
    (r'\bdata\s+loss\b', 'data_risk', 0.30), (r'\bcrash\w*\b', 'system_failure', 0.22),
    (r'\bnot\s+(loading|working|responding)\b', 'functional_failure', 0.20),
    (r'\blocke?d\s+out\b', 'access_blocked', 0.24), (r'\bpayment\s+fail\w*\b', 'payment_failure', 0.22),
    (r'\bcompromised\b', 'account_risk', 0.28), (r'\bimmediately\b', 'urgency', 0.14),
    (r'\burgent\b', 'urgency', 0.14), (r'\bransomware\b', 'security_threat', 0.40),
]

DEESC_EV = [
    (r'\bhow\s+do\s+i\b', 'informational', -0.14), (r'\bwhere\s+is\b', 'informational', -0.14),
    (r'\bfeature\s+request\b', 'feature_req', -0.16), (r'\bheadquarters\b', 'general_query', -0.20),
    (r'\broadmap\b', 'general_query', -0.16),
]

RT_BENCH = {
    ('Fraud', 'Critical'): 4, ('Fraud', 'High'): 12, ('Technical', 'Critical'): 5,
    ('Technical', 'High'): 18, ('Technical', 'Medium'): 38, ('Technical', 'Low'): 50,
    ('Billing', 'Critical'): 6, ('Billing', 'High'): 20, ('Billing', 'Medium'): 42,
    ('Billing', 'Low'): 52, ('Account', 'High'): 22, ('Account', 'Medium'): 40,
    ('Account', 'Low'): 50, ('General Inquiry', 'Medium'): 35, ('General Inquiry', 'Low'): 45,
}

CAT_SEV = {
    'Fraud':           {'exp': 'Critical', 'w': 0.28,  'note': 'Fraud carries inherent security risk'},
    'Technical':       {'exp': 'High',     'w': 0.18,  'note': 'Technical failures impact availability'},
    'Account':         {'exp': 'Medium',   'w': 0.12,  'note': 'Account issues affect user access'},
    'Billing':         {'exp': 'Medium',   'w': 0.10,  'note': 'Billing issues have financial impact'},
    'General Inquiry': {'exp': 'Low',      'w': -0.15, 'note': 'General inquiries are informational'},
}

# --- Helper Functions ---
def rt_tier(h):
    if h <= 10: return 'FAST'
    if h <= 45: return 'MID'
    return 'SLOW'

def make_input(row):
    rt = float(row.get('Resolution_Time_Hours', 30))
    return (f"[SUBJ] {row['Ticket_Subject']} [BODY] {row['Ticket_Description']} "
            f"| cat:{row['Issue_Category']} | ch:{row['Ticket_Channel']} "
            f"| rt:{rt_tier(rt)} | pri:{row['Priority_Level']}")

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    best = MDL / 'best'
    if not best.exists():
        return None, None, 0.5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok    = AutoTokenizer.from_pretrained(str(best))
    base   = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2, ignore_mismatched_sizes=True)
    model  = PeftModel.from_pretrained(base, str(best)).float().to(device)
    model.eval()
    tf  = MDL / 'threshold.npy'
    thr = float(np.load(str(tf))[0]) if tf.exists() else 0.5
    return tok, model, thr

def predict(texts, tok, model, thr):
    device = next(model.parameters()).device
    probs  = []
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i+32], truncation=True, padding='max_length',
                  max_length=256, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        probs += torch.softmax(out.logits.float(), -1)[:, 1].cpu().tolist()
    probs = np.array(probs)
    return probs, (probs >= thr).astype(int)

def dir_score(row):
    t  = f"{row['Ticket_Subject']} {row['Ticket_Description']}".lower()
    s  = sum(w for _, _, w in ESCA_EV + DEESC_EV if re.search(_, t))
    cs = CAT_SEV.get(row['Issue_Category'], {'exp': 'Medium', 'w': 0.05})
    if PMAP.get(cs['exp'], 1) > PMAP.get(row['Priority_Level'], 1): s += abs(cs['w'])
    elif PMAP.get(cs['exp'], 1) < PMAP.get(row['Priority_Level'], 1): s -= abs(cs['w'])
    sat = int(row['Satisfaction_Score'])
    if sat <= 2 and row['Priority_Level'] in ('Low', 'Medium'): s += 0.18
    elif sat >= 4 and row['Priority_Level'] in ('Critical', 'High'): s -= 0.12
    rt = float(row['Resolution_Time_Hours'])
    exp = RT_BENCH.get((row['Issue_Category'], row['Priority_Level']), 40.0)
    r = rt / max(exp, 1)
    if r < 0.4: s += 0.14
    elif r > 2.5: s += 0.10
    return s

def get_verdict(row, prob):
    base = PMAP.get(row['Priority_Level'], 1)
    d = dir_score(row)
    if d < 0:
        bump = 2 if prob >= 0.90 else 1
        mtype = 'False Alarm'
        inf = max(0, base - bump)
    else:
        bump = 2 if prob >= 0.85 else 1
        mtype = 'Hidden Crisis'
        inf = min(3, base + bump)
    return RPMAP[inf], mtype

def make_dossier(row, prob):
    inf, mtype = get_verdict(row, prob)
    t = f"{row['Ticket_Subject']} {row['Ticket_Description']}".lower()
    ev = []
    for pat, etype, w in ESCA_EV + DEESC_EV:
        m = re.search(pat, t)
        if m and (w > 0) == (mtype == 'Hidden Crisis'):
            fld = 'Ticket_Subject' if re.search(pat, row['Ticket_Subject'].lower()) else 'Ticket_Description'
            ev.append({'signal': 'keyword', 'type': etype, 'value': m.group(0), 'source_field': fld, 'weight': round(w, 3)})
    
    rt = float(row['Resolution_Time_Hours'])
    exp = RT_BENCH.get((row['Issue_Category'], row['Priority_Level']), 40.0)
    r = rt / max(exp, 1)
    ev.append({'signal': 'resolution_time', 'value': f"{rt:.0f}h", 'expected': f"~{exp:.0f}h",
               'source_field': 'Resolution_Time_Hours', 'weight': round(0.25 if r < 0.4 else 0.20 if r > 2.5 else 0.05, 3)})
    
    cs = CAT_SEV.get(row['Issue_Category'], {'exp': 'Medium', 'w': 0.05, 'note': ''})
    ev.append({'signal': 'category_baseline', 'value': row['Issue_Category'],
               'source_field': 'Issue_Category', 'weight': round(abs(cs['w']), 3)})
    
    ev.append({'signal': 'satisfaction_score', 'value': str(int(row['Satisfaction_Score'])),
               'source_field': 'Satisfaction_Score', 'weight': round(0.18 if int(row['Satisfaction_Score']) <= 2 else 0.02, 3)})
    
    ev = sorted(ev, key=lambda x: abs(x.get('weight', 0)), reverse=True)
    
    a = PMAP.get(row['Priority_Level'], 1)
    ii = PMAP.get(inf, 1)
    d = ii - a
    delta = (f"+{d} (under-prioritised by {d} level{'s' if d > 1 else ''})" if d > 0
             else f"{d} (over-prioritised by {abs(d)} level{'s' if abs(d) > 1 else ''})" if d < 0
             else "0 (borderline)")
    
    kp = [e for e in ev if e['signal'] == 'keyword' and e.get('weight', 0) > 0]
    kn = [e for e in ev if e['signal'] == 'keyword' and e.get('weight', 0) < 0]
    
    s1 = f"This {row['Issue_Category']} ticket via {row['Ticket_Channel']} assigned {row['Priority_Level']} — model infers {inf}."
    s2 = (f"Escalation indicators ({', '.join(repr(e['value']) for e in kp[:2])}) signal higher severity." if mtype == 'Hidden Crisis' and kp
          else f"Low-severity indicators suggest {row['Priority_Level']} is over-assigned." if mtype == 'False Alarm' and kn
          else "Semantic patterns and metadata signal priority mismatch.")
    s3 = (f"RT={rt:.0f}h with sat={int(row['Satisfaction_Score'])}/5 supports under-prioritisation." if mtype == 'Hidden Crisis'
          else f"RT={rt:.0f}h with sat={int(row['Satisfaction_Score'])}/5 consistent with over-triage.")
    
    return {'ticket_id': str(row.get('Ticket_ID', '')), 'assigned_priority': row['Priority_Level'],
            'inferred_severity': inf, 'mismatch_type': mtype, 'severity_delta': delta,
            'confidence': round(float(prob), 4), 'feature_evidence': ev, 'constraint_analysis': f"{s1} {s2} {s3}"}

# --- Main App Logic ---
tok, model, THR = load_model()
model_ready = tok is not None

st.markdown("<h1 style='color:#e05c2a'>🔍 SIA — Support Integrity Auditor</h1>", unsafe_allow_html=True)
st.caption("Detect priority mismatches in CRM support tickets · MARS Open Projects 2026")

if not model_ready:
    st.error("⚠️ Model not found at `models/sia_model/best`. Please ensure you have run the training pipeline first.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🎫 Single Ticket", "📂 Batch Upload", "📊 Dashboard"])

# --- Tab 1: Single Ticket ---
with tab1:
    st.subheader("Analyse a Single Ticket")
    c1, c2 = st.columns(2)
    with c1:
        subject  = st.text_input("Ticket Subject", value="Cannot access my account")
        category = st.selectbox("Issue Category", ['Technical', 'Fraud', 'Account', 'Billing', 'General Inquiry'])
        channel  = st.selectbox("Ticket Channel", ['Email', 'Chat', 'Phone', 'Web Form', 'Social Media'])
        priority = st.selectbox("Assigned Priority", ['Low', 'Medium', 'High', 'Critical'])
    with c2:
        desc = st.text_area("Ticket Description", height=130,
                            value="I have been locked out of my account since yesterday. I cannot access any of my data and payment is overdue.")
        rt   = st.number_input("Resolution Time (hours)", min_value=0.0, value=48.0, step=1.0)
        sat  = st.slider("Satisfaction Score", 1, 5, 2)

    if st.button("🔍 Analyse Ticket", type="primary"):
        row = {'Ticket_ID': 'SINGLE-001', 'Ticket_Subject': subject, 'Ticket_Description': desc,
               'Issue_Category': category, 'Ticket_Channel': channel, 'Priority_Level': priority,
               'Resolution_Time_Hours': rt, 'Satisfaction_Score': sat}
        txt   = make_input(row)
        probs, preds = predict([txt], tok, model, THR)
        prob  = probs[0]
        pred  = preds[0]

        st.divider()
        col_a, col_b, col_c = st.columns(3)
        if pred == 1:
            inf, mtype = get_verdict(row, prob)
            col_a.metric("Verdict", "🚨 MISMATCH", mtype)
            col_b.metric("Confidence", f"{prob:.1%}")
            col_c.metric("Inferred Severity", inf, f"Assigned: {priority}")
            st.error(f"**{mtype}** — Assigned `{priority}` but model infers `{inf}`")
            dos = make_dossier(row, prob)
            with st.expander("📋 Evidence Dossier", expanded=True):
                st.json(dos)
        else:
            col_a.metric("Verdict", "✅ CONSISTENT")
            col_b.metric("Confidence (Mismatch)", f"{prob:.1%}")
            col_c.metric("Priority", priority)
            st.success(f"Priority **{priority}** appears correctly assigned. (confidence mismatch: {prob:.1%})")

# --- Tab 2: Batch Upload ---
with tab2:
    st.subheader("Batch Analysis")
    uploaded = st.file_uploader("Upload CSV (same columns as training data)", type="csv")
    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.info(f"Loaded {len(df_up):,} tickets")
        req   = ['Ticket_ID', 'Ticket_Subject', 'Ticket_Description', 'Issue_Category',
                 'Priority_Level', 'Ticket_Channel', 'Resolution_Time_Hours', 'Satisfaction_Score']
        miss  = [c for c in req if c not in df_up.columns]
        if miss:
            st.error(f"Missing columns: {miss}")
        else:
            if st.button("🚀 Run Analysis", type="primary"):
                with st.spinner("Running inference…"):
                    for c in ['Ticket_Subject', 'Ticket_Description', 'Issue_Category', 'Ticket_Channel', 'Priority_Level']:
                        df_up[c] = df_up[c].astype(str)
                    texts            = df_up.apply(make_input, axis=1).tolist()
                    prbs, pds        = predict(texts, tok, model, THR)
                    df_up['prob']    = prbs
                    df_up['pred']    = pds
                    df_up['verdict'] = df_up['pred'].map({0: 'Consistent', 1: 'Mismatch'})

                n = df_up['pred'].sum()
                st.success(f"✅ Done — {n:,}/{len(df_up):,} mismatches flagged ({n/len(df_up)*100:.1f}%)")

                flagged = df_up[df_up['pred'] == 1].copy()
                verdicts = []
                for _, r in flagged.iterrows():
                    inf, mt = get_verdict(r.to_dict(), r['prob'])
                    verdicts.append({'ticket_id': str(r['Ticket_ID']), 'mtype': mt, 'inferred': inf})
                
                if len(verdicts) > 0:
                    vdf = pd.DataFrame(verdicts)
                    flagged = flagged.reset_index(drop=True).join(vdf.set_index('ticket_id'), on='Ticket_ID')

                st.dataframe(
                    df_up[['Ticket_ID', 'Priority_Level', 'Issue_Category', 'verdict', 'prob']].style
                    .apply(lambda r: ['background:#ffe0e0' if r['verdict'] == 'Mismatch' else '' for _ in r], axis=1),
                    use_container_width=True
                )

                csv_out = df_up.to_csv(index=False).encode()
                st.download_button("⬇️ Download Predictions CSV", csv_out, "predictions.csv", "text/csv")

                dos = [make_dossier(r.to_dict(), r['prob']) for _, r in flagged.iterrows()]
                st.download_button("⬇️ Download Dossiers JSON",
                                   json.dumps(dos, indent=2).encode(),
                                   "evidence_dossiers.json", "application/json")

                st.session_state['batch_df'] = df_up

# --- Tab 3: Dashboard ---
with tab3:
    st.subheader("Priority Mismatch Dashboard")

    src = None
    if 'batch_df' in st.session_state:
        src = st.session_state['batch_df']
    else:
        lab = Path("outputs/labeled_tickets.csv")
        if lab.exists():
            src = pd.read_csv(lab)
            if 'predicted' not in src.columns and 'label' in src.columns:
                src['predicted'] = src['label']
            if 'verdict' not in src.columns:
                src['verdict'] = src['predicted'].map({0: 'Consistent', 1: 'Mismatch'})
            if 'mtype' not in src.columns and 'mismatch_type' in src.columns:
                src['mtype'] = src['mismatch_type']

    if src is None:
        st.info("Run a Batch Analysis or train the model first to populate the dashboard.")
    else:
        n_tot = len(src)
        n_mis = src['predicted'].sum() if 'predicted' in src.columns else 0
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Tickets", f"{n_tot:,}")
        m2.metric("Flagged Mismatches", f"{n_mis:,}", f"{n_mis/n_tot*100:.1f}%" if n_tot > 0 else "0%")
        hc = src[src.get('mtype', '') == 'Hidden Crisis'].shape[0] if 'mtype' in src.columns else 0
        m3.metric("Hidden Crisis", f"{hc:,}")

        st.divider()
        r1c1, r1c2 = st.columns(2)

        with r1c1:
            st.markdown("**Mismatch Type Distribution**")
            if 'mtype' in src.columns:
                vc = src[src['mtype'].isin(['Hidden Crisis', 'False Alarm', 'Consistent'])]['mtype'].value_counts()
                if len(vc) > 0:
                    fig = px.pie(values=vc.values, names=vc.index,
                                 color_discrete_map={'Hidden Crisis': '#e05c2a', 'False Alarm': '#5b8fc9', 'Consistent': '#5e9e6e'},
                                 hole=0.4)
                    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280)
                    st.plotly_chart(fig, use_container_width=True)

        with r1c2:
            st.markdown("**Mismatch Rate by Category**")
            if 'predicted' in src.columns:
                grp = src.groupby('Issue_Category')['predicted'].mean().sort_values(ascending=True) * 100
                fig = px.bar(x=grp.values, y=grp.index, orientation='h',
                             color=grp.values, color_continuous_scale='RdYlGn_r',
                             labels={'x': 'Mismatch Rate (%)', 'y': ''})
                fig.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10), height=280)
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("**Severity Delta Heatmap — Category × Channel**")
        if 'delta' in src.columns:
            pivot = src.pivot_table(values='delta', index='Issue_Category',
                                    columns='Ticket_Channel', aggfunc='mean').fillna(0)
            fig = px.imshow(pivot, color_continuous_scale='RdBu_r', color_continuous_midpoint=0,
                            text_auto='.2f', aspect='auto',
                            labels={'color': 'Mean Δ', 'x': 'Channel', 'y': 'Category'})
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280)
            st.plotly_chart(fig, use_container_width=True)
        elif 'severity_delta' in src.columns or 'inferred_ord' in src.columns:
            st.info("Severity delta computed during training. Upload labeled_tickets.csv to see heatmap.")

        st.divider()
        st.markdown("**Top Contributing Signal Types**")
        dos_path = Path("outputs/evidence_dossiers.json")
        if dos_path.exists():
            with open(dos_path) as f:
                dos_data = json.load(f)
            sig_counts = {}
            for d in dos_data:
                for ev in d.get('feature_evidence', []):
                    sig = ev.get('type') or ev.get('signal', 'unknown')
                    sig_counts[sig] = sig_counts.get(sig, 0) + 1
            if sig_counts:
                sc_df = pd.DataFrame(list(sig_counts.items()), columns=['Signal Type', 'Count'])
                sc_df = sc_df.sort_values('Count', ascending=True).tail(10)
                fig = px.bar(sc_df, x='Count', y='Signal Type', orientation='h',
                             color='Count', color_continuous_scale='Blues')
                fig.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10), height=300)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run predict.py on a batch to populate signal distribution.")

st.divider()
st.caption("SIA · MARS Open Projects 2026 · DeBERTa-v3-small + LoRA · 4-signal self-supervised pipeline")
