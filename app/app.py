"""
Run with: streamlit run app/app.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from ml.predictor import predict, get_all_categories

#Page Config 
st.set_page_config(
    page_title="NNHire — Job Category Predictor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

#Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .nn-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        color: white;
    }

    .nn-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }

    .nn-header p {
        font-size: 0.95rem;
        opacity: 0.75;
        margin: 0;
    }

    .nn-badge {
        display: inline-block;
        background: #e94560;
        color: white;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1px;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
    }

    .result-card {
        background: #f8faff;
        border: 1.5px solid #e0e7ff;
        border-left: 5px solid #4f46e5;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }

    .result-card .label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }

    .result-card .value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e1b4b;
    }

    .conf-high {
        color: #059669;
        font-weight: 700;
    }

    .conf-medium {
        color: #d97706;
        font-weight: 700;
    }

    .conf-low {
        color: #dc2626;
        font-weight: 700;
    }

    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    .sidebar-note {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.82rem;
        color: #475569;
        margin-top: 1rem;
    }

    /* Button */
    .stButton > button {
        background: #4f46e5;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background: #3730a3;
        color: white;
    }

    /* Input */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1.5px solid #e0e7ff;
        font-size: 1rem;
        padding: 0.6rem 1rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #4f46e5;
        box-shadow: 0 0 0 3px rgba(79,70,229,0.1);
    }

    /* Section divider */
    .section-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #9ca3af;
        margin: 1.5rem 0 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Session State 
if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts: {title, predicted, final, confidence}

# Header 
st.markdown("""
<div class="nn-header">
    <div class="nn-badge">NovaNectar Services</div>
    <h1>NNHire — Job Category Predictor</h1>
    <p>Enter a job title and the ML model predicts its category instantly. Edit if needed before confirming.</p>
</div>
""", unsafe_allow_html=True)

#Sidebar
with st.sidebar:
    st.markdown("###  About the Model")
    st.markdown("""
    | Detail | Value |
    |--------|-------|
    | Algorithm | Logistic Regression |
    | Features | TF-IDF (1–2 grams) |
    | Input | Job Title |
    | Output | Job Category |
    """)
    st.markdown("""
    <div class="sidebar-note">
                 The model weights the job title 3× more than the description for better accuracy on short inputs.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Supported Categories")
    try:
        cats = get_all_categories()
        for c in cats:
            st.markdown(f"- {c}")
    except Exception:
        st.info("Train the model first to see categories.")

    st.markdown("---")

# Main Panel
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    st.markdown('<div class="section-title">Step 1 — Enter Job Title</div>', unsafe_allow_html=True)

    job_title = st.text_input(
        "Job Title",
        placeholder="e.g. React Developer, HR Recruiter, Data Scientist...",
        label_visibility="collapsed",
    )

    predict_btn = st.button(" Predict Category", use_container_width=True)

    #Prediction 
    if predict_btn and job_title.strip():
        try:
            result = predict(job_title.strip())

            predicted_cat  = result["category"]
            confidence     = result["confidence"]
            all_scores     = result["all_scores"]

            # Confidence color class
            if confidence >= 70:
                conf_class = "conf-high"
                conf_emoji = "🟢"
            elif confidence >= 40:
                conf_class = "conf-medium"
                conf_emoji = "🟡"
            else:
                conf_class = "conf-low"
                conf_emoji = "🔴"

            # Store in session so the edit widget can reference it
            st.session_state["last_result"] = {
                "title"     : job_title.strip(),
                "category"  : predicted_cat,
                "confidence": confidence,
                "all_scores": all_scores,
            }

        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

    #Show Result & Edit 
    if "last_result" in st.session_state:
        r = st.session_state["last_result"]

        conf = r["confidence"]
        conf_class = "conf-high" if conf >= 70 else ("conf-medium" if conf >= 40 else "conf-low")
        conf_emoji = "🟢" if conf >= 70 else ("🟡" if conf >= 40 else "🔴")

        st.markdown(f"""
        <div class="result-card">
            <div class="label">Predicted Category</div>
            <div class="value">{r['category']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f"{conf_emoji} Confidence: <span class='{conf_class}'>{conf:.1f}%</span>",
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-title">Step 2 — Edit if Needed</div>', unsafe_allow_html=True)

        all_cats = get_all_categories()
        default_idx = all_cats.index(r["category"]) if r["category"] in all_cats else 0
        final_category = st.selectbox(
            "Final Category",
            options=all_cats,
            index=default_idx,
            label_visibility="collapsed",
        )

        confirm_btn = st.button("Confirm & Save", use_container_width=True)
        if confirm_btn:
            st.session_state.history.append({
                "Job Title"        : r["title"],
                "Predicted"        : r["category"],
                "Final Category"   : final_category,
                "Confidence (%)"   : f"{conf:.1f}",
                "Edited?"          : "Yes" if final_category != r["category"] else "No",
            })
            st.success(f"Saved: **{r['title']}** → **{final_category}**")
            del st.session_state["last_result"]

    elif not (predict_btn and job_title.strip()):
        st.info("👆 Enter a job title above and click Predict.")

with col_right:
    st.markdown('<div class="section-title">Confidence Breakdown</div>', unsafe_allow_html=True)

    if "last_result" in st.session_state:
        r = st.session_state["last_result"]
        scores = r["all_scores"]

        # Show top 6 as a bar chart
        top6 = dict(list(scores.items())[:6])
        chart_df = pd.DataFrame({
            "Category"      : list(top6.keys()),
            "Confidence (%)" : list(top6.values()),
        })
        st.bar_chart(chart_df.set_index("Category"), color="#4f46e5")
    else:
        st.markdown("""
        <div style="
            background: #f8faff;
            border: 1.5px dashed #c7d2fe;
            border-radius: 10px;
            padding: 3rem 1rem;
            text-align: center;
            color: #a5b4fc;
            font-size: 0.9rem;
        ">
            Confidence chart appears here<br>after prediction
        </div>
        """, unsafe_allow_html=True)

# ─── Submission History ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("###Submission History")

if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True, hide_index=True)

    # Download as CSV
    csv = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download History as CSV",
        data=csv,
        file_name="nnhire_submissions.csv",
        mime="text/csv",
    )
else:
    st.markdown("""
    <div style="
        background: #f8faff;
        border: 1.5px dashed #e0e7ff;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        color: #9ca3af;
        font-size: 0.88rem;
    ">
        No submissions yet. Predictions you confirm will appear here.
    </div>
    """, unsafe_allow_html=True)

#Footer
st.markdown("""
<br>
<div style="text-align:center; font-size:0.78rem; color:#9ca3af;">
    NNHire &nbsp;·&nbsp; NovaNectar Services Pvt. Ltd. &nbsp;·&nbsp;
    ML: TF-IDF + Logistic Regression &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
