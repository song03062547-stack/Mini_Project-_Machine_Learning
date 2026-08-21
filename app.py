"""
app.py
------
Streamlit App: Apple Quality Prediction ("Orchard Lab" theme)
ทำนายคุณภาพแอปเปิล (good/bad) จากค่าคุณสมบัติต่าง ๆ ของผลไม้
ใช้โมเดลที่เทรนไว้แล้วจาก train.py

รันแบบ local: streamlit run app.py

ผู้จัดทำ: นายปฐมพงศ์ ชัยสรรค์  รหัสนักศึกษา 664245039
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Orchard Lab | Apple Quality Predictor",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = Path("model")
DATA_PATH = Path("data/apple_quality.csv")

AUTHOR_NAME = "นายปฐมพงศ์ ชัยสรรค์"
AUTHOR_ID = "664245039"
AUTHOR_ADDRESS = "66/44"

# =========================================================================
# THEME — "Orchard Lab": โทนไม้/สวนผลไม้ยามเช้า สมุดบันทึกงานวิจัยเกษตร
# Palette:  cream #FAF6EC | orchard green #23412F | leaf #6F9A3C
#           apple red #B3432B | gold ripeness #D8A93B | ink #1E2A20
# Type:     Fraunces (display, serif มีเอกลักษณ์) + Inter (body/UI)
# =========================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

:root{
    --cream:      #FAF6EC;
    --cream-2:    #F1EADA;
    --orchard:    #23412F;
    --orchard-2:  #2F5540;
    --leaf:       #6F9A3C;
    --apple-red:  #B3432B;
    --gold:       #D8A93B;
    --ink:        #1E2A20;
    --ink-soft:   #4B5A4F;
    --line:       #D8CFB8;
}

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(1200px 500px at 100% -10%, #EFE6CC 0%, transparent 60%),
        var(--cream);
    color: var(--ink);
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--orchard) 0%, #1A3324 100%);
    border-right: 1px solid #14261A;
}
section[data-testid="stSidebar"] * { color: #F1EADA !important; }
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stMarkdown p { color: #E4DDC6 !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }

section[data-testid="stSidebar"] .stButton button {
    background: var(--gold);
    color: var(--orchard) !important;
    border: none;
    border-radius: 999px;
    font-weight: 700;
    padding: 0.6rem 1rem;
    letter-spacing: 0.02em;
    box-shadow: 0 6px 16px rgba(0,0,0,0.25);
    transition: transform .15s ease, box-shadow .15s ease;
}
section[data-testid="stSidebar"] .stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.32);
}
section[data-testid="stSidebar"] .stButton button p { color: var(--orchard) !important; }

/* ---------- Headings ---------- */
h1, h2, h3, .hero-title {
    font-family: 'Fraunces', serif !important;
    color: var(--orchard) !important;
    letter-spacing: -0.01em;
}

/* ---------- Hero banner ---------- */
.hero-wrap{
    display:flex; align-items:center; justify-content:space-between; gap: 24px;
    background: linear-gradient(120deg, var(--orchard) 0%, var(--orchard-2) 55%, #3C6B4C 100%);
    border-radius: 20px;
    padding: 34px 40px;
    margin-bottom: 22px;
    box-shadow: 0 14px 30px rgba(35,65,47,0.28);
    position: relative;
    overflow: hidden;
}
.hero-wrap::after{
    content:"";
    position:absolute; right:-40px; top:-60px;
    width:220px; height:220px; border-radius:50%;
    background: radial-gradient(circle, rgba(216,169,59,0.35) 0%, transparent 70%);
}
.hero-eyebrow{
    font-family:'Inter'; font-weight:700; font-size:0.72rem; letter-spacing:0.16em;
    text-transform:uppercase; color: var(--gold); margin-bottom:6px;
}
.hero-title{
    color:#FAF6EC !important; font-size:2.3rem; font-weight:700; margin:0; line-height:1.15;
}
.hero-sub{
    color:#DCE6DC; font-size:0.98rem; margin-top:10px; max-width:560px;
}
.hero-badges{ display:flex; gap:10px; margin-top:16px; flex-wrap:wrap; }
.badge{
    background: rgba(250,246,236,0.12);
    border: 1px solid rgba(250,246,236,0.28);
    color:#F1EADA; padding: 6px 14px; border-radius:999px;
    font-size:0.8rem; font-weight:600;
}
.badge b{ color: var(--gold); }
.hero-emoji{ font-size: 5.2rem; filter: drop-shadow(0 10px 18px rgba(0,0,0,0.35)); }

/* ---------- Cards ---------- */
.crate-card{
    background: #FFFEF9;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 6px 18px rgba(35,65,47,0.06);
}
.section-label{
    font-weight:700; font-size:0.75rem; letter-spacing:0.12em; text-transform:uppercase;
    color: var(--leaf); margin-bottom: 6px;
}

.result-good, .result-bad{
    border-radius:16px; padding: 22px 24px; margin-bottom: 14px;
    font-family:'Fraunces', serif; font-weight:600; font-size:1.35rem;
    display:flex; align-items:center; gap:14px;
}
.result-good{
    background: linear-gradient(120deg, #E9F3DE, #DCEBCB);
    border: 1px solid #B9D89A; color: #2F4A1B;
}
.result-bad{
    background: linear-gradient(120deg, #F7E4DD, #F1D3C7);
    border: 1px solid #E3AC97; color: #7A2E1B;
}
.result-emoji{ font-size: 2.1rem; }
.result-caption{
    font-family:'Inter'; font-weight:500; font-size:0.9rem; opacity:0.85; margin-top:2px;
}

/* ---------- Metric-style stat cards ---------- */
.stat-card{
    background:#FFFEF9; border:1px solid var(--line); border-radius:14px;
    padding: 16px 18px; text-align:left;
}
.stat-num{ font-family:'Fraunces',serif; font-size:1.7rem; font-weight:700; color:var(--orchard); }
.stat-label{ font-size:0.78rem; color:var(--ink-soft); font-weight:600; text-transform:uppercase; letter-spacing:0.06em; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"]{ gap: 6px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"]{
    background: transparent; border-radius: 10px 10px 0 0; padding: 10px 18px;
    color: var(--ink-soft); font-weight:600;
}
.stTabs [aria-selected="true"]{
    background: var(--cream-2) !important; color: var(--orchard) !important;
    border-bottom: 3px solid var(--leaf) !important;
}

/* ---------- Misc ---------- */
hr{ border-color: var(--line); }
.footer-wrap{
    margin-top: 30px; padding: 22px 26px; border-radius:16px;
    background: linear-gradient(120deg, var(--orchard) 0%, #1E3627 100%);
    color:#EFE9D6; display:flex; justify-content:space-between; align-items:center;
    flex-wrap:wrap; gap: 10px;
    box-shadow: 0 10px 24px rgba(35,65,47,0.25);
}
.footer-wrap .who{ font-weight:700; font-family:'Fraunces', serif; font-size:1.05rem; color:#FAF6EC; }
.footer-wrap .meta{ font-size:0.85rem; color:#C9D6C7; }
.footer-wrap .tag{
    background: rgba(216,169,59,0.18); border:1px solid rgba(216,169,59,0.4);
    color: var(--gold); padding:4px 12px; border-radius:999px; font-size:0.78rem; font-weight:700;
}

::-webkit-scrollbar{ width:10px; }
::-webkit-scrollbar-thumb{ background: var(--line); border-radius: 10px; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "apple_quality_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    le = joblib.load(MODEL_DIR / "label_encoder.pkl")
    feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
    meta = joblib.load(MODEL_DIR / "meta.pkl")
    return model, scaler, le, feature_cols, meta


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["A_id"]).copy()
    df["Acidity"] = pd.to_numeric(df["Acidity"], errors="coerce")
    df = df.drop(columns=["A_id"]).dropna().reset_index(drop=True)
    return df


model, scaler, le, feature_cols, meta = load_artifacts()
df = load_data()

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#FFFEF9",
    plot_bgcolor="#FFFEF9",
    font=dict(family="Inter, sans-serif", color="#1E2A20"),
    colorway=["#6F9A3C", "#B3432B", "#D8A93B", "#23412F"],
    margin=dict(l=10, r=10, t=40, b=10),
)

# =========================================================================
# Sidebar
# =========================================================================
st.sidebar.markdown(
    """
    <div style="text-align:center; padding: 6px 0 14px 0;">
        <div style="font-size:2.6rem; line-height:1;">🍎</div>
        <div style="font-family:'Fraunces',serif; font-weight:700; font-size:1.25rem; margin-top:6px;">Orchard Lab</div>
        <div style="font-size:0.78rem; color:#C9D6C7; letter-spacing:0.06em; text-transform:uppercase;">Apple Quality Studio</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "ปรับค่าคุณสมบัติของแอปเปิลด้านล่าง แล้วให้โมเดลทำนายว่า "
    "แอปเปิลลูกนั้นคุณภาพ **ดี (good)** หรือ **ไม่ดี (bad)**"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**🎚️ ปรับค่าคุณสมบัติแอปเปิล**")

input_values = {}
feature_help = {
    "Size": "ขนาดของผล (ค่ามาตรฐาน, ยิ่งมากยิ่งใหญ่)",
    "Weight": "น้ำหนักของผล (ค่ามาตรฐาน)",
    "Sweetness": "ระดับความหวาน",
    "Crunchiness": "ความกรอบของเนื้อ",
    "Juiciness": "ความฉ่ำน้ำ",
    "Ripeness": "ระดับความสุก",
    "Acidity": "ความเป็นกรด",
}
feature_icon = {
    "Size": "📏", "Weight": "⚖️", "Sweetness": "🍯", "Crunchiness": "🥢",
    "Juiciness": "💧", "Ripeness": "🌞", "Acidity": "🍋",
}

for col in feature_cols:
    lo, hi = meta["feature_ranges"][col]
    default = float(np.round((lo + hi) / 2, 2))
    input_values[col] = st.sidebar.slider(
        f"{feature_icon.get(col, '•')} {col}",
        min_value=float(np.floor(lo)),
        max_value=float(np.ceil(hi)),
        value=default,
        step=0.1,
        help=feature_help.get(col, ""),
    )

predict_btn = st.sidebar.button("🔮  ทำนายคุณภาพ", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div style="font-size:0.78rem; color:#C9D6C7; line-height:1.6;">
        <b style="color:#F1EADA;">ผู้จัดทำ</b><br/>
        {AUTHOR_NAME}<br/>
        รหัสนักศึกษา {AUTHOR_ID}<br/>
        {AUTHOR_ADDRESS}
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================================
# Hero
# =========================================================================
st.markdown(
    f"""
    <div class="hero-wrap">
        <div>
            <div class="hero-eyebrow">Mini Machine Learning Project</div>
            <div class="hero-title">Orchard Lab — Apple Quality Predictor</div>
            <div class="hero-sub">
                ทำนายคุณภาพแอปเปิลจากคุณสมบัติทางกายภาพของผล 7 ตัวแปร
                ด้วยโมเดล {meta['best_model_name']} ที่ผ่านการเปรียบเทียบ
                และคัดเลือกจากหลายอัลกอริทึม
            </div>
            <div class="hero-badges">
                <span class="badge">🎯 Accuracy <b>{meta['test_accuracy']*100:.1f}%</b></span>
                <span class="badge">🧪 Samples <b>{len(df):,}</b></span>
                <span class="badge">📦 Model <b>{meta['best_model_name']}</b></span>
            </div>
        </div>
        <div class="hero-emoji">🍎</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["🔮  ทำนายผล", "📊  สำรวจข้อมูล (EDA)", "🧠  เกี่ยวกับโมเดล"])

# ----- Tab 1: Prediction -----
with tab1:
    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.markdown('<div class="crate-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">ค่าคุณสมบัติที่เลือก</div>', unsafe_allow_html=True)
        input_df = pd.DataFrame([input_values])[feature_cols]
        st.dataframe(input_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="crate-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">ผลการทำนาย</div>', unsafe_allow_html=True)

        if predict_btn:
            X_scaled = scaler.transform(input_df)
            pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            label = le.inverse_transform([pred])[0]
            good_idx = list(le.classes_).index("good")
            good_prob = proba[good_idx] * 100

            if label == "good":
                st.markdown(
                    f"""
                    <div class="result-good">
                        <div class="result-emoji">🍏</div>
                        <div>
                            แอปเปิลลูกนี้คุณภาพ <u>ดี</u> (good)
                            <div class="result-caption">ความมั่นใจของโมเดล {good_prob:.1f}%</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-bad">
                        <div class="result-emoji">🍎</div>
                        <div>
                            แอปเปิลลูกนี้คุณภาพ <u>ไม่ดี</u> (bad)
                            <div class="result-caption">ความมั่นใจว่าเป็น good เพียง {good_prob:.1f}%</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=good_prob,
                    number={"suffix": "%", "font": {"color": "#23412F", "family": "Fraunces"}},
                    title={"text": "โอกาสที่จะเป็นแอปเปิลคุณภาพดี", "font": {"size": 14}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#4B5A4F"},
                        "bar": {"color": "#6F9A3C" if good_prob >= 50 else "#B3432B"},
                        "bgcolor": "#FFFEF9",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 50], "color": "#F7E4DD"},
                            {"range": [50, 100], "color": "#E9F3DE"},
                        ],
                        "threshold": {
                            "line": {"color": "#D8A93B", "width": 4},
                            "thickness": 0.85,
                            "value": good_prob,
                        },
                    },
                )
            )
            fig.update_layout(
                height=280,
                margin=dict(l=20, r=20, t=50, b=10),
                paper_bgcolor="#FFFEF9",
                font=dict(family="Inter, sans-serif"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 ปรับค่าที่แถบด้านซ้ายมือ แล้วกดปุ่ม **ทำนายคุณภาพ** ได้เลย")
        st.markdown("</div>", unsafe_allow_html=True)

# ----- Tab 2: EDA -----
with tab2:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="stat-card"><div class="stat-num">{len(df):,}</div>'
            f'<div class="stat-label">ตัวอย่างทั้งหมด</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="stat-card"><div class="stat-num">{len(feature_cols)}</div>'
            f'<div class="stat-label">Feature ที่ใช้</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        good_pct = (df["Quality"] == "good").mean() * 100
        st.markdown(
            f'<div class="stat-card"><div class="stat-num">{good_pct:.1f}% : {100-good_pct:.1f}%</div>'
            f'<div class="stat-label">สัดส่วน Good : Bad</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="crate-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">การกระจายตัวของแต่ละ Feature แยกตามคุณภาพ</div>', unsafe_allow_html=True)
    feat_choice = st.selectbox("เลือก Feature", feature_cols, label_visibility="collapsed")
    fig_hist = px.histogram(
        df, x=feat_choice, color="Quality", barmode="overlay", nbins=40,
        color_discrete_map={"good": "#6F9A3C", "bad": "#B3432B"},
    )
    fig_hist.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="crate-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">ความสัมพันธ์ระหว่าง Feature (Correlation Heatmap)</div>', unsafe_allow_html=True)
    corr = df[feature_cols].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1
    )
    fig_corr.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="crate-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">ตัวอย่างข้อมูลดิบ</div>', unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----- Tab 3: About model -----
with tab3:
    st.markdown('<div class="crate-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">รายละเอียดโมเดล</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f'<div class="stat-card"><div class="stat-num">{meta["best_model_name"]}</div>'
            f'<div class="stat-label">โมเดลที่ใช้</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="stat-card"><div class="stat-num">{meta["test_accuracy"]*100:.2f}%</div>'
            f'<div class="stat-label">Test Accuracy</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="stat-card"><div class="stat-num">{meta["cv_accuracy_mean"]*100:.2f}%</div>'
            f'<div class="stat-label">CV Accuracy เฉลี่ย</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    if hasattr(model, "feature_importances_"):
        st.markdown('<div class="crate-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">ความสำคัญของแต่ละ Feature</div>', unsafe_allow_html=True)
        fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(
            ascending=True
        )
        fig_fi = px.bar(
            fi, orientation="h", labels={"value": "Importance", "index": "Feature"},
            color=fi.values, color_continuous_scale=["#D8A93B", "#6F9A3C", "#23412F"],
        )
        fig_fi.update_layout(**PLOTLY_LAYOUT, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="crate-card">', unsafe_allow_html=True)
    st.markdown(
        """
<div class="section-label">Pipeline การทำงาน</div>

1. โหลดข้อมูลจาก `apple_quality.csv` และทำความสะอาด (ตัดแถว metadata, แปลงชนิดข้อมูล)
2. Scale ข้อมูลด้วย `StandardScaler`
3. เทรนและเปรียบเทียบ 3 โมเดล: Logistic Regression, Random Forest, SVM
4. เลือกโมเดลที่แม่นยำที่สุดจาก Test set แล้วบันทึกไว้
5. โหลดโมเดลมาใช้ทำนายแบบ real-time ผ่านหน้าเว็บ Streamlit นี้
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================================
# Footer
# =========================================================================
st.markdown(
    f"""
    <div class="footer-wrap">
        <div>
            <div class="who">{AUTHOR_NAME}</div>
            <div class="meta">รหัสนักศึกษา {AUTHOR_ID} · {AUTHOR_ADDRESS}</div>
        </div>
        <div class="tag">Mini ML Project • scikit-learn + Streamlit</div>
    </div>
    """,
    unsafe_allow_html=True,
)
