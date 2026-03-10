import streamlit as st
import base64
from logic import VideoSummarizerLogic
import time

# --- Configuration ---
st.set_page_config(
    page_title="AI Video Notes Architect",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper functions ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_密=True)

# --- Custom Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #E2E8F0;
    }

    /* Main App Container Glassmorphism */
    .stApp {
        background-color: transparent;
    }

    /* Title Animation */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #F0ABFC, #818CF8, #2DD4BF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        letter-spacing: -2px;
        animation: float 6s ease-in-out infinite;
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-bottom: 3rem;
        font-weight: 300;
    }

    /* Input Field Styling with Glow */
    .stTextInput > div > div > input {
        background-color: rgba(15, 23, 42, 0.6) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 16px !important;
        padding: 1.2rem !important;
        backdrop-filter: blur(20px) !important;
        transition: all 0.4s ease !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #818CF8 !important;
        box-shadow: 0 0 20px rgba(129, 140, 248, 0.3) !important;
        transform: scale(1.01);
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #7C3AED, #2563EB) !important;
        color: white !important;
        border: none !important;
        padding: 1rem 2rem !important;
        border-radius: 16px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        width: 100% !important;
        text-transform: uppercase;
        font-size: 0.9rem !important;
    }

    .stButton > button:hover {
        transform: translateY(-5px) scale(1.02) !important;
        box-shadow: 0 20px 25px -5px rgba(124, 58, 237, 0.4) !important;
    }

    /* Perfect Insight Container */
    .perfect-insight {
        background: rgba(30, 41, 59, 0.4);
        padding: 2.5rem;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 6px solid #818CF8;
        backdrop-filter: blur(40px);
        margin: 2rem 0;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }

    .insight-label {
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 0.75rem;
        color: #818CF8;
        font-weight: 800;
        margin-bottom: 1rem;
    }

    /* Detailed Note Cards */
    .note-card {
        background: rgba(15, 23, 42, 0.3);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.03);
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    .note-card:hover {
        background: rgba(30, 41, 59, 0.5);
        transform: translateX(10px);
        border-color: rgba(129, 140, 248, 0.3);
    }

    /* Sidebar Refinement */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.85) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #7C3AED, #2DD4BF) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- App Context ---
summarizer = VideoSummarizerLogic()

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric/512/artificial-intelligence.png", width=80)
    st.markdown("### <span style='color: #818CF8'>ENGINE CONFIG</span>", unsafe_allow_html=True)
    
    model_options = {
        "Small (Fastest, CPU Optimized)": "google/flan-t5-small",
        "Base (High Quality, Slower)": "google/flan-t5-base"
    }
    model_friendly_name = st.selectbox("LLM Architecture", list(model_options.keys()), index=0)
    model_choice = model_options[model_friendly_name]
    
    chunk_size = st.slider("Context Window Size", 500, 2000, 1200)
    st.divider()
    st.markdown("### ⚡ SPEED OPTIMIZATIONS")
    st.success("Parallel Orchestration")
    st.success("Greedy Decoding")
    st.divider()
    st.markdown("### WORKFLOW")
    st.info("""
    1. **Scrape** Transcript
    2. **Orchestrate** Chunks
    3. **Synthesize** (Parallel)
    4. **Revise** for Insight
    """)

# --- Main UI ---
st.markdown('<h1 class="main-title">Notes Architect</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Transform hours of video into perfect actionable insights instantly.</p>', unsafe_allow_html=True)

url = st.text_input("YouTube Video URL", placeholder="Paste transmission link here...")

if st.button("Initialize Analysis"):
    if url:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(val, text):
            progress_bar.progress(val)
            status_text.text(text.upper())
            
        with st.spinner("Decoding transmission..."):
            start_time = time.time()
            fragmented_notes, master_insight, error = summarizer.process_video(url, model_name=model_choice, progress_callback=update_progress)
            end_time = time.time()
            
            if error:
                st.error(error)
            else:
                st.balloons()
                st.success(f"SYSTEM ANALYZED CONTENT IN {end_time - start_time:.1f} SECONDS")
                st.divider()

                # --- High-Level Perfect Insight ---
                st.markdown(f'''
                <div class="perfect-insight">
                    <div class="insight-label">✨ PERFECT INSIGHT (REVISED)</div>
                    <div style="font-size: 1.25rem; font-weight: 300; letter-spacing: 0.5px;">{master_insight}</div>
                </div>
                ''', unsafe_allow_html=True)

                # --- Detailed Segment Notes ---
                with st.expander("🔍 VIEW ALL SYSTEM NOTES", expanded=False):
                    for note in fragmented_notes:
                        st.markdown(f'''
                        <div class="note-card">
                            {note}
                        </div>
                        ''', unsafe_allow_html=True)
                
                # Export
                full_notes = f"# Perfect Insight\n\n{master_insight}\n\n## Detailed Notes\n\n" + "\n".join([f"- {n}" for n in fragmented_notes])
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.download_button(
                        label="Download Report",
                        data=full_notes,
                        file_name="architect_insight.md",
                        mime="text/markdown"
                    )
    else:
        st.warning("Please provide a valid transmission source.")

# --- Background ---
try:
    set_png_as_page_bg("C:\\Users\\hp\\.gemini\\antigravity\\brain\\daf69a7c-1853-4e96-9ec9-636db5579f21\\ultra_premium_ai_bg_1773090701831.png")
except:
    pass
