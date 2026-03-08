import streamlit as st
import pandas as pd
import json
import re
import os

# ─── System prompt: forces JSON-structured EDA output ───
SYSTEM_PROMPT = """You are an expert data analyst. When given CSV data, perform a thorough Exploratory Data Analysis (EDA) and return your findings as a JSON object.

Your JSON response must follow this exact structure:
{
  "summary": {
    "totalRows": number,
    "totalColumns": number,
    "overview": "2-3 sentence plain-English overview of what this dataset contains"
  },
  "columns": [
    {
      "name": "column name",
      "type": "numeric|categorical|text|date",
      "nullCount": number,
      "uniqueCount": number,
      "insight": "1 sentence insight about this column"
    }
  ],
  "statistics": [
    {
      "column": "column name (numeric columns only)",
      "min": number,
      "max": number,
      "mean": number,
      "median": number,
      "stdDev": number
    }
  ],
  "topInsights": [
    "Insight 1 as a clear, actionable sentence",
    "Insight 2",
    "Insight 3",
    "Insight 4",
    "Insight 5"
  ],
  "categoryBreakdowns": [
    {
      "column": "categorical column name",
      "values": [{"label": "value", "count": number}]
    }
  ],
  "anomalies": ["Any anomaly or data quality issue found"],
  "recommendations": ["Actionable recommendation 1", "Actionable recommendation 2", "Actionable recommendation 3"]
}

Rules:
- Return ONLY valid JSON, no markdown, no backticks, no extra text
- Analyze ALL columns thoroughly
- For numeric columns always compute min, max, mean, median, stdDev
- For categorical columns always compute categoryBreakdowns (top 5 values with counts)
- topInsights must be the most valuable business or analytical findings
- anomalies should flag nulls, outliers, duplicates, or suspicious patterns
- recommendations must be specific and actionable based on the data"""


# ─── LLM provider helpers ───
def get_llm(provider: str, model: str, api_key: str = ""):
    """Return a LangChain LLM instance based on selected provider."""
    if provider == "Ollama (Local)":
        from langchain_community.llms import Ollama
        return Ollama(model=model)
    elif provider == "Groq (Cloud – Free)":
        from langchain_groq import ChatGroq
        return ChatGroq(model_name=model, groq_api_key=api_key, temperature=0)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def invoke_llm(llm, prompt: str, provider: str) -> str:
    """Invoke the LLM and return text, handling both LLM and ChatModel."""
    if provider == "Groq (Cloud – Free)":
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    else:
        return llm.invoke(prompt)


# ─── Page config ───
st.set_page_config(page_title="AI Data Analyst", layout="wide", page_icon="📊")


# ─── Custom CSS for premium look ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 2rem; }

    /* Hero header */
    .hero-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem; letter-spacing: -0.5px;
    }
    .hero-subtitle { color: #8892a4; font-size: 1.05rem; font-weight: 400; margin-bottom: 1.5rem; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(248,250,252,0.9));
        border: 1px solid rgba(102, 126, 234, 0.15); border-radius: 16px;
        padding: 1.4rem 1.6rem; text-align: center;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15); }
    .metric-value {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.2;
    }
    .metric-label {
        font-size: 0.82rem; color: #8892a4; font-weight: 600;
        text-transform: uppercase; letter-spacing: 1px; margin-top: 0.3rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem; font-weight: 700; color: #1e293b;
        margin: 2rem 0 1rem 0; padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.2);
        display: flex; align-items: center; gap: 0.5rem;
    }

    /* Insight cards */
    .insight-card {
        background: linear-gradient(135deg, #f0f4ff 0%, #faf5ff 100%);
        border-left: 4px solid #667eea; border-radius: 0 12px 12px 0;
        padding: 1rem 1.2rem; margin-bottom: 0.7rem;
        font-size: 0.92rem; color: #334155; line-height: 1.6; transition: background 0.2s;
    }
    .insight-card:hover { background: linear-gradient(135deg, #e8eeff 0%, #f3ecff 100%); }

    /* Anomaly badges */
    .anomaly-badge {
        background: linear-gradient(135deg, #fff7ed, #fef3c7);
        border-left: 4px solid #f59e0b; border-radius: 0 12px 12px 0;
        padding: 0.9rem 1.2rem; margin-bottom: 0.6rem;
        font-size: 0.9rem; color: #92400e; line-height: 1.5;
    }

    /* Recommendation cards */
    .rec-card {
        background: linear-gradient(135deg, #ecfdf5, #f0fdf4);
        border-left: 4px solid #10b981; border-radius: 0 12px 12px 0;
        padding: 0.9rem 1.2rem; margin-bottom: 0.6rem;
        font-size: 0.9rem; color: #065f46; line-height: 1.5;
    }

    /* Stat table */
    .stat-table {
        width: 100%; border-collapse: separate; border-spacing: 0;
        border-radius: 12px; overflow: hidden;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06); font-size: 0.88rem;
    }
    .stat-table th {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; padding: 0.8rem 1rem; text-align: left;
        font-weight: 600; letter-spacing: 0.3px;
    }
    .stat-table td { padding: 0.7rem 1rem; border-bottom: 1px solid #f1f5f9; color: #334155; }
    .stat-table tr:nth-child(even) td { background: #f8fafc; }
    .stat-table tr:hover td { background: #f0f4ff; }

    /* Column pill */
    .col-pill {
        display: inline-block; padding: 0.25rem 0.7rem; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.3px;
    }
    .col-pill.numeric     { background: #dbeafe; color: #1e40af; }
    .col-pill.categorical { background: #ede9fe; color: #5b21b6; }
    .col-pill.text        { background: #fce7f3; color: #9d174d; }
    .col-pill.date        { background: #d1fae5; color: #065f46; }

    /* Category bar */
    .cat-bar-container { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.4rem; }
    .cat-bar-label { min-width: 100px; text-align: right; font-size: 0.82rem; color: #475569; font-weight: 500; }
    .cat-bar-track { flex: 1; height: 24px; background: #f1f5f9; border-radius: 12px; overflow: hidden; }
    .cat-bar-fill {
        height: 100%; border-radius: 12px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        display: flex; align-items: center; justify-content: flex-end;
        padding-right: 8px; color: white; font-size: 0.72rem; font-weight: 700;
        min-width: 24px; transition: width 0.6s ease;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #f8f9ff 0%, #f0f2ff 100%); }
    [data-testid="stSidebar"] h1 { font-size: 1.2rem; }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important; border: none !important;
        border-radius: 12px !important; padding: 0.7rem 2rem !important;
        font-weight: 600 !important; font-size: 0.95rem !important;
        letter-spacing: 0.3px !important;
        transition: transform 0.15s, box-shadow 0.15s !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(102,126,234,0.35) !important;
    }
    .stButton > button:active { transform: translateY(0px) !important; }

    /* Divider */
    .section-divider {
        height: 1px; background: linear-gradient(90deg, transparent, rgba(102,126,234,0.2), transparent);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Header ───
st.markdown('<div class="hero-title">📊 AI Data Analyst Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Upload a CSV file · Get structured EDA insights powered by AI · 100 % private with local models</div>', unsafe_allow_html=True)


# ─── Sidebar ───
with st.sidebar:
    st.markdown("## 🗂️ Data Upload")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    st.markdown("---")
    st.markdown("### ⚙️ AI Provider")

    provider = st.radio(
        "Choose backend",
        ["Ollama (Local)", "Groq (Cloud – Free)"],
        index=0,
        help="Ollama runs on your machine. Groq is a free cloud API — get a key at console.groq.com",
    )

    if provider == "Ollama (Local)":
        model_name = st.selectbox("Ollama model", ["llama3", "mistral", "phi3", "qwen2.5-coder:7b"], index=0)
        api_key = ""
    else:
        model_name = st.selectbox("Groq model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it", "mixtral-8x7b-32768"], index=0)
        # Try Streamlit secrets first (for deployed app), then text input
        default_key = ""
        try:
            default_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            default_key = os.environ.get("GROQ_API_KEY", "")
        api_key = st.text_input("Groq API Key", value=default_key, type="password",
                                help="Free at https://console.groq.com/keys")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#94a3b8;font-size:0.78rem;margin-top:1rem;'>"
        "Powered by Ollama / Groq<br>Your data stays private"
        "</div>",
        unsafe_allow_html=True,
    )


# ─── Helper: Try to parse JSON from LLM response ───
def parse_llm_json(raw_text: str) -> dict | None:
    """Extract and parse JSON from the LLM response, tolerating stray text."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text).strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def section(emoji: str, title: str):
    st.markdown(f'<div class="section-header">{emoji} {title}</div>', unsafe_allow_html=True)


# ─── Main flow ───
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ── Data preview ──
    section("📋", "Data Preview")
    col_info, col_table = st.columns([1, 3])
    with col_info:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{df.shape[0]:,}</div>'
            f'<div class="metric-label">Rows</div></div>', unsafe_allow_html=True)
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{df.shape[1]}</div>'
            f'<div class="metric-label">Columns</div></div>', unsafe_allow_html=True)
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        nulls = int(df.isnull().sum().sum())
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{nulls}</div>'
            f'<div class="metric-label">Missing Values</div></div>', unsafe_allow_html=True)
    with col_table:
        st.dataframe(df.head(15), use_container_width=True, height=340)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Run analysis ──
    can_run = True
    if provider == "Groq (Cloud – Free)" and not api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar to continue.")
        can_run = False

    if can_run and st.button("🚀  Analyze Dataset", use_container_width=True):
        csv_text = df.to_csv(index=False)
        max_chars = 8000
        if len(csv_text) > max_chars:
            csv_text = csv_text[:max_chars] + "\n... (truncated)"

        user_prompt = f"Analyze this CSV data and return full EDA insights in the required JSON format:\n\n{csv_text}"
        full_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt

        with st.spinner("🔍 Analyzing your data with AI…"):
            try:
                llm = get_llm(provider, model_name, api_key)
                raw_response = invoke_llm(llm, full_prompt, provider)
            except Exception as e:
                st.error(f"❌ LLM error: {e}")
                raw_response = None

        if raw_response:
            result = parse_llm_json(raw_response)
            if result is None:
                st.error("⚠️ The model returned non-JSON output. Raw response shown below.")
                st.code(raw_response, language="json")
            else:
                st.session_state["eda_result"] = result
                st.session_state["eda_raw"] = raw_response

    # ─── Render results if available ───
    if "eda_result" in st.session_state:
        result = st.session_state["eda_result"]

        # ── Summary ──
        summary = result.get("summary", {})
        if summary:
            section("🔎", "Dataset Summary")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">{summary.get("totalRows", "—"):,}</div>'
                    f'<div class="metric-label">Total Rows</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">{summary.get("totalColumns", "—")}</div>'
                    f'<div class="metric-label">Total Columns</div></div>', unsafe_allow_html=True)
            with c3:
                total_nulls = sum(c.get("nullCount", 0) for c in result.get("columns", []))
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">{total_nulls}</div>'
                    f'<div class="metric-label">Total Nulls</div></div>', unsafe_allow_html=True)
            st.markdown(
                f"<p style='color:#475569;font-size:0.95rem;margin-top:1rem;line-height:1.7;'>"
                f"{summary.get('overview', '')}</p>", unsafe_allow_html=True)

        # ── Column Analysis ──
        columns_data = result.get("columns", [])
        if columns_data:
            section("🏛️", "Column Analysis")
            rows_html = ""
            for col in columns_data:
                col_type = col.get("type", "text")
                pill_class = col_type if col_type in ("numeric", "categorical", "text", "date") else "text"
                rows_html += (
                    f"<tr>"
                    f"<td style='font-weight:600'>{col.get('name', '')}</td>"
                    f"<td><span class='col-pill {pill_class}'>{col_type.upper()}</span></td>"
                    f"<td style='text-align:center'>{col.get('nullCount', 0)}</td>"
                    f"<td style='text-align:center'>{col.get('uniqueCount', '—')}</td>"
                    f"<td>{col.get('insight', '')}</td></tr>"
                )
            st.markdown(
                f"""<table class="stat-table"><thead><tr>
                    <th>Column</th><th>Type</th><th style='text-align:center'>Nulls</th>
                    <th style='text-align:center'>Unique</th><th>Insight</th>
                </tr></thead><tbody>{rows_html}</tbody></table>""", unsafe_allow_html=True)

        # ── Statistics ──
        stats = result.get("statistics", [])
        if stats:
            section("📐", "Numeric Statistics")
            rows_html = ""
            for s in stats:
                fmt = lambda v: f"{v:,.2f}" if isinstance(v, (int, float)) else str(v)
                rows_html += (
                    f"<tr><td style='font-weight:600'>{s.get('column', '')}</td>"
                    f"<td style='text-align:right'>{fmt(s.get('min', ''))}</td>"
                    f"<td style='text-align:right'>{fmt(s.get('max', ''))}</td>"
                    f"<td style='text-align:right'>{fmt(s.get('mean', ''))}</td>"
                    f"<td style='text-align:right'>{fmt(s.get('median', ''))}</td>"
                    f"<td style='text-align:right'>{fmt(s.get('stdDev', ''))}</td></tr>"
                )
            st.markdown(
                f"""<table class="stat-table"><thead><tr>
                    <th>Column</th><th style='text-align:right'>Min</th><th style='text-align:right'>Max</th>
                    <th style='text-align:right'>Mean</th><th style='text-align:right'>Median</th>
                    <th style='text-align:right'>Std Dev</th>
                </tr></thead><tbody>{rows_html}</tbody></table>""", unsafe_allow_html=True)

        # ── Top Insights ──
        insights = result.get("topInsights", [])
        if insights:
            section("💡", "Top Insights")
            for i, insight in enumerate(insights, 1):
                st.markdown(
                    f'<div class="insight-card"><strong>#{i}</strong> &nbsp; {insight}</div>',
                    unsafe_allow_html=True)

        # ── Category Breakdowns ──
        cats = result.get("categoryBreakdowns", [])
        if cats:
            section("📊", "Category Breakdowns")
            cols = st.columns(min(len(cats), 3))
            for idx, cat in enumerate(cats):
                with cols[idx % 3]:
                    st.markdown(f"**{cat.get('column', '')}**")
                    values = cat.get("values", [])
                    if values:
                        max_count = max(v.get("count", 1) for v in values) or 1
                        for v in values:
                            pct = (v.get("count", 0) / max_count) * 100
                            st.markdown(
                                f"""<div class="cat-bar-container">
                                    <div class="cat-bar-label">{v.get("label", "")}</div>
                                    <div class="cat-bar-track">
                                        <div class="cat-bar-fill" style="width:{pct}%">{v.get("count", 0)}</div>
                                    </div></div>""", unsafe_allow_html=True)
                    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # ── Anomalies ──
        anomalies = result.get("anomalies", [])
        if anomalies:
            section("⚠️", "Anomalies & Data Quality")
            for a in anomalies:
                st.markdown(f'<div class="anomaly-badge">🔸 {a}</div>', unsafe_allow_html=True)

        # ── Recommendations ──
        recs = result.get("recommendations", [])
        if recs:
            section("✅", "Recommendations")
            for r in recs:
                st.markdown(f'<div class="rec-card">→ {r}</div>', unsafe_allow_html=True)

        # ── Raw JSON ──
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        with st.expander("🔧 View Raw JSON Response"):
            st.json(result)

else:
    st.markdown(
        """<div style="text-align:center; margin-top:6rem; color:#94a3b8;">
            <div style="font-size:4rem; margin-bottom:1rem;">📁</div>
            <div style="font-size:1.2rem; font-weight:600; color:#64748b;">No dataset uploaded yet</div>
            <div style="font-size:0.92rem; margin-top:0.5rem;">
                Use the sidebar to upload a CSV file and start your analysis
            </div></div>""", unsafe_allow_html=True)
