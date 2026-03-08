import streamlit as st
import pandas as pd
import json
import re
import os

# ─── System prompt: forces JSON-structured EDA output ───
SYSTEM_PROMPT = """You are an expert Amazon sales data analyst. When given CSV data, analyze it thoroughly and return your findings as a strictly structured JSON object.

Your JSON response must follow this EXACT structure:

{
  "summary": {
    "totalRows": number,
    "totalColumns": number,
    "overview": "2-3 sentence plain-English overview of what this dataset contains"
  },

  "summaryCards": [
    { "metric": "Total Rows", "value": number },
    { "metric": "Total Columns", "value": number },
    { "metric": "Numeric Columns", "value": number },
    { "metric": "Insights Found", "value": number }
  ],

  "topInsights": [
    {
      "number": 1,
      "bold": "Short bold headline of the insight",
      "detail": "Full sentence explanation with specific numbers and column names"
    },
    {
      "number": 2,
      "bold": "Short bold headline",
      "detail": "Full sentence explanation"
    },
    {
      "number": 3,
      "bold": "Short bold headline",
      "detail": "Full sentence explanation"
    },
    {
      "number": 4,
      "bold": "Short bold headline",
      "detail": "Full sentence explanation"
    },
    {
      "number": 5,
      "bold": "Short bold headline",
      "detail": "Full sentence explanation"
    }
  ],

  "columns": [
    {
      "name": "exact column name from CSV",
      "type": "numeric|categorical|text|date",
      "nullCount": number,
      "uniqueCount": number,
      "insight": "One sentence describing what this column contains and its distribution"
    }
  ],

  "statistics": [
    {
      "column": "numeric column name",
      "min": number,
      "max": number,
      "mean": number,
      "median": number,
      "stdDev": number
    }
  ],

  "categoryBreakdowns": [
    {
      "column": "categorical column name",
      "values": [
        { "label": "category value", "count": number }
      ]
    }
  ],

  "anomalies": [
    "Anomaly 1 described as a full sentence with specific numbers",
    "Anomaly 2 described as a full sentence with specific numbers",
    "Anomaly 3 described as a full sentence with specific numbers",
    "Anomaly 4 described as a full sentence with specific numbers"
  ],

  "recommendations": [
    {
      "number": 1,
      "bold": "Short action headline",
      "detail": "Full sentence explaining what to do and why, referencing specific data findings"
    },
    {
      "number": 2,
      "bold": "Short action headline",
      "detail": "Full sentence explanation"
    },
    {
      "number": 3,
      "bold": "Short action headline",
      "detail": "Full sentence explanation"
    }
  ]
}

STRICT RULES:
- Return ONLY valid JSON — no markdown, no backticks, no extra text before or after
- Analyze every single column in the CSV without skipping any
- For every numeric column compute: min, max, mean, median, stdDev rounded to 2 decimal places
- For every categorical column compute: top 5 values with exact counts in categoryBreakdowns
- topInsights must reference specific column names, product names, and real numbers from the data
- anomalies must flag: zero-value rows, missing data, outliers, duplicate patterns, suspicious distributions
- recommendations must be specific and directly reference findings from the data — no generic advice
- median and mean for sparse/zero-heavy columns should reflect the true distribution including zeros
- All monetary values should include the currency symbol found in the data (e.g. ₹)"""


# ─── LLM provider helpers ───
def get_llm(provider: str, model: str, api_key: str = ""):
    """Return a LangChain LLM instance based on selected provider."""
    if provider == "Ollama (Local)":
        from langchain_community.chat_models import ChatOllama
        # Ensure model name matches what's pulled on the machine
        m = model if ":" in model else f"{model}:latest"
        return ChatOllama(model=m, temperature=0)
    elif provider == "Groq (Cloud – Free)":
        from langchain_groq import ChatGroq
        return ChatGroq(model_name=model, groq_api_key=api_key, temperature=0)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def invoke_llm(llm, prompt: str, system_prompt: str, provider: str) -> str:
    """Invoke the LLM with system context and return text contents."""
    from langchain_core.messages import SystemMessage, HumanMessage
    
    # All our models are now used as ChatModels for better prompt following
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)


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


    # ─── LLM Config (Background) ───
    # Auto-detect Groq key, otherwise default to Ollama
    api_key = ""
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")

    if api_key:
        provider = "Groq (Cloud – Free)"
        model_name = "llama-3.3-70b-versatile"
    else:
        provider = "Ollama (Local)"
        model_name = "llama3:latest"



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
        st.warning("⚠️ Groq API key not found in backend configuration. Please add it to `.streamlit/secrets.toml`.")
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
                # Attempt primary provider
                llm = get_llm(provider, model_name, api_key)
                raw_response = invoke_llm(llm, user_prompt, SYSTEM_PROMPT, provider)
            except Exception as e:
                error_msg = str(e)
                # Fallback if Groq key is invalid
                if provider == "Groq (Cloud – Free)" and ("401" in error_msg or "invalid_api_key" in error_msg.lower()):
                    st.warning("⚠️ Groq API key is invalid or expired. Falling back to local Ollama...")
                    try:
                        fallback_provider = "Ollama (Local)"
                        fallback_model = "llama3:latest"
                        llm = get_llm(fallback_provider, fallback_model, "")
                        raw_response = invoke_llm(llm, user_prompt, SYSTEM_PROMPT, fallback_provider)
                    except Exception as e_fallback:
                        st.error(f"❌ Both Groq and Ollama fallback failed. Error: {e_fallback}")
                        raw_response = None
                else:
                    # Specific hint for Ollama local errors
                    if "connection" in error_msg.lower() and provider == "Ollama (Local)":
                        error_msg += "\n\n💡 TIP: Make sure Ollama is open and running on your taskbar!"
                    st.error(f"❌ LLM error: {error_msg}")
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

        # ── Summary & Metrics ──
        summary = result.get("summary", {})
        if summary or result.get("summaryCards"):
            section("🔎", "Dataset Summary")
            
            # Priority 1: Use summaryCards if provided
            summary_cards = result.get("summaryCards", [])
            if summary_cards:
                cols = st.columns(len(summary_cards))
                for idx, card in enumerate(summary_cards):
                    metric = card.get("metric", "Metric")
                    val = card.get("value", "—")
                    # Format numbers with commas if they are numeric
                    if isinstance(val, (int, float)):
                        val_str = f"{val:,}" if val >= 1000 else str(val)
                    else:
                        val_str = str(val)
                    with cols[idx]:
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-value">{val_str}</div>'
                            f'<div class="metric-label">{metric}</div></div>', unsafe_allow_html=True)
            else:
                # Fallback for old structure
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
            
            if summary.get('overview'):
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
                if isinstance(insight, dict):
                    bold = insight.get("bold", "")
                    detail = insight.get("detail", "")
                    st.markdown(
                        f'<div class="insight-card"><strong>#{i} {bold}</strong><br>{detail}</div>',
                        unsafe_allow_html=True)
                else:
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
            for i, r in enumerate(recs, 1):
                if isinstance(r, dict):
                    bold = r.get("bold", "")
                    detail = r.get("detail", "")
                    st.markdown(
                        f'<div class="rec-card"><strong>{i}. {bold}</strong><br>{detail}</div>',
                        unsafe_allow_html=True)
                else:
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
