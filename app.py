import os
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv, set_key

from src.extractor import extract_text_from_pdf
from src.parser import extract_questions_via_llm
from src.classifier import classify_all
from src.reporter import generate_reports

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Qunify AI - Automated Exam Parser & Classifier",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Premium Modern Glassmorphism Theme (AuraQ / Qunify)
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
    /* Main Layout */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0f0f16 70%, #08070b 100%) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #f1f5f9 !important;
    }
    
    /* Headers & Titles */
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #4f46e5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    /* Glassmorphism Cards */
    div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
        background: rgba(30, 41, 59, 0.25);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
    }
    
    /* Interactive Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 0.6rem 1.8rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.45) !important;
        background: linear-gradient(135deg, #818cf8 0%, #4f46e5 100%) !important;
    }
    
    /* Form inputs and text fields */
    .stTextInput>div>div>input, .stTextArea>div>textarea, .stSelectbox>div>div {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
    }
    
    /* Upload Box custom layout */
    div[data-testid="stFileUploaderDropzone"] {
        background: rgba(30, 41, 59, 0.15) !important;
        border: 2px dashed rgba(99, 102, 241, 0.3) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        transition: border 0.3s ease;
    }
    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #6366f1 !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0c0a0f !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: rgba(15, 23, 42, 0.3);
        padding: 6px 12px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background-color: transparent;
        color: #94a3b8;
        font-size: 15px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        color: #c7d2fe !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #6366f1 !important;
    }
</style>
""", unsafe_allow_html=True)

# Custom Banner Component
st.markdown("""
<div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(79, 70, 229, 0.03) 100%); padding: 2rem; border-radius: 16px; border: 1px solid rgba(99, 102, 241, 0.15); margin-bottom: 2rem; display: flex; align-items: center; justify-content: space-between;">
    <div>
        <h1 style="margin: 0; font-size: 2.8rem; font-weight: 800;">⚡ Qunify AI</h1>
        <p style="margin: 0.5rem 0 0 0; color: #94a3b8; font-size: 1.1rem; font-weight: 400; letter-spacing: 0.01em;">
            Automated Exam Parser & Intelligent Question Classification Agent
        </p>
    </div>
    <div style="font-size: 3.5rem; filter: drop-shadow(0 0 12px rgba(99, 102, 241, 0.4));">🚀</div>
</div>
""", unsafe_allow_html=True)

# Initialize Session State
if "questions" not in st.session_state:
    st.session_state.questions = []
if "classifications" not in st.session_state:
    st.session_state.classifications = []
if "categories" not in st.session_state:
    st.session_state.categories = []

# Sidebar Config
st.sidebar.markdown("""
<div style="padding-bottom: 1rem; border-bottom: 1px solid rgba(255, 255, 255, 0.05); margin-bottom: 1.5rem;">
    <h3 style="margin: 0; font-size: 1.3rem;">⚙️ Settings Panel</h3>
</div>
""", unsafe_allow_html=True)

# API Key Config
default_api_key = os.environ.get("GEMINI_API_KEY", "")
api_key = st.sidebar.text_input("Gemini API Key", value=default_api_key, type="password")
if api_key != default_api_key:
    os.environ["GEMINI_API_KEY"] = api_key
    set_key(".env", "GEMINI_API_KEY", api_key)

# Model Selection Dropdowns
st.sidebar.subheader("🤖 Agent Models")
available_models = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-3.1-pro-preview"
]

extraction_model = st.sidebar.selectbox(
    "1. Question Extraction Model",
    options=available_models,
    index=0,
    help="Select which Gemini model to use for scanning raw text and extracting clean question blocks."
)

classification_model = st.sidebar.selectbox(
    "2. Topic Classification Model",
    options=available_models,
    index=0,
    help="Select which Gemini model to use for categorising your extracted questions."
)

st.sidebar.subheader("🎛️ Optimization Tuning")
chunk_size = st.sidebar.slider(
    "Pages per Extraction Chunk",
    min_value=3,
    max_value=20,
    value=5,
    step=1,
    help="Higher values reduce the number of API requests during parsing. Default is 5."
)

batch_size = st.sidebar.slider(
    "Questions per Classification Batch",
    min_value=10,
    max_value=150,
    value=25,
    step=5,
    help="Consolidate more questions into a single call. Setting this to 150 will classify the entire paper in a single request!"
)

max_questions = st.sidebar.number_input(
    "Expected Questions Count",
    min_value=10,
    max_value=300,
    value=150,
    step=5,
    help="Define the exact number of questions in this paper (e.g. 150 for core subject, 100 for General Studies)."
)

# Category Configuration
st.sidebar.subheader("📋 Classification Categories")
categories_input = st.sidebar.text_area(
    "Enter categories (one per line):",
    value="Materials Science\nEDC\nAnalog Circuits\nNetwork Theory\nControl Systems\nElectromagnetic Theory\nMeasurements\nCommunication Systems\nAdvance Communications\nAdvance Electronics\nBasic Electrical Engineering\nComputer Organization\nSignals and Systems\nDigital Circuits\nMicroprocessors"
)

st.session_state.categories = [c.strip() for c in categories_input.split("\n") if c.strip()]

# File Uploader Container
st.markdown("### 📥 Step 1: Upload Exam Paper")
uploaded_file = st.file_uploader("", type=["pdf", "txt"])

if uploaded_file is not None:
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.success(f"Successfully loaded: {uploaded_file.name}")
    
    # Process Button
    if st.button("⚡ Run Extraction & Classification Agent"):
        # Clear previous state to prevent stale data display during execution
        st.session_state.questions = []
        st.session_state.classifications = []
        
        from src.logger import setup_logger
        run_logger = setup_logger("run.log")
        
        try:
            with st.status("Analyzing Exam PDF...", expanded=True) as status:
                # 1. Extraction / Cache Check
                status.update(label="Extracting text from PDF...")
                
                def ocr_callback(msg):
                    status.write(f"📖 {msg}")
                
                # If uploaded file is already a text file, bypass extractor
                if file_path.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        raw_text = f.read()
                    # Create a dummy pdf next to it for parser caching logic
                    dummy_pdf = file_path.replace('.txt', '.pdf')
                    # Cache the txt content so extractor module loads it as cached
                    os.makedirs("cache", exist_ok=True)
                    cached_txt_path = os.path.join("cache", f"txt_{os.path.splitext(os.path.basename(dummy_pdf))[0]}.txt")
                    with open(cached_txt_path, "w", encoding="utf-8") as f:
                        f.write(raw_text)
                    status.write("📖 Loaded text file directly.")
                else:
                    raw_text = extract_text_from_pdf(file_path, cache_dir="cache", progress_callback=ocr_callback)
                    
                # 2. Parsing (LLM-based)
                status.update(label="Using AI to parse and extract clean question blocks...")
                
                def parse_callback(msg):
                    status.write(f"🔍 {msg}")
                    
                st.session_state.questions = extract_questions_via_llm(
                    raw_text, 
                    max_q=max_questions,
                    model_name=extraction_model, 
                    chunk_size=chunk_size,
                    progress_callback=parse_callback
                )
                status.write(f"🎯 Parsed {len(st.session_state.questions)} questions successfully.")
                
                # 3. Classifying
                status.update(label="Batching questions to Gemini API...")
                
                def classification_callback(msg):
                    status.write(f"🧠 {msg}")
                
                # Temporary write categories to file for the classification runner
                temp_categories_path = "temp_categories.csv"
                with open(temp_categories_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(st.session_state.categories))
                    
                classifications = classify_all(
                    st.session_state.questions, 
                    temp_categories_path, 
                    model_name=classification_model,
                    batch_size=batch_size,
                    progress_callback=classification_callback
                )
                
                if classifications:
                    st.session_state.classifications = classifications
                    st.write(f"Classified {len(classifications)} questions.")
                    status.update(label="Classification Complete!", state="complete")
                else:
                    status.update(label="Classification failed. Please check Gemini API logs.", state="error")
        finally:
            run_logger.close()

# Display Results & Verification Grid
if st.session_state.questions and st.session_state.classifications:
    st.markdown("---")
    st.subheader("🔍 Step 2: Interactive Verification & Exports Workspace")
    st.info("Verify classifications, manually adjust categories, and preview files directly in the tabs below.")
    
    # Map classifications into a DataFrame structure
    combined_data = []
    class_map = {}
    for c in st.session_state.classifications:
        if isinstance(c, dict):
            q_id = c.get('id', c.get('Id', c.get('q_num')))
            q_cat = c.get('category', c.get('Category', 'Unclassified'))
            if q_id is not None:
                class_map[q_id] = q_cat
        elif hasattr(c, 'id'):
            class_map[c.id] = getattr(c, 'category', 'Unclassified')
        
    for q in st.session_state.questions:
        q_id = q['id']
        assigned_cat = class_map.get(q_id, "Unclassified")
        combined_data.append({
            "Question Number": q_id,
            "Snippet": q["snippet"],
            "Assigned Category": assigned_cat
        })
        
    df = pd.DataFrame(combined_data)
    
    # Create tab interface
    tab_editor, tab_markdown, tab_text = st.tabs([
        "📋 Questions Editor Grid", 
        "📝 Markdown Summary Preview", 
        "📄 Raw Mappings (Text)"
    ])
    
    with tab_editor:
        st.subheader("Edit Question Classifications")
        edited_df = st.data_editor(
            df,
            column_config={
                "Question Number": st.column_config.NumberColumn("Q. No.", disabled=True),
                "Snippet": st.column_config.TextColumn("Snippet Preview", width="large", disabled=True),
                "Assigned Category": st.column_config.SelectboxColumn(
                    "Category Selection",
                    options=st.session_state.categories + ["Unclassified"],
                    width="large"
                )
            },
            hide_index=True,
            use_container_width=True
        )
        
    # Generate content for preview tabs
    final_classifications = []
    for index, row in edited_df.iterrows():
        final_classifications.append({
            'id': int(row["Question Number"]),
            'category': row["Assigned Category"]
        })
        
    # Generate temporary reports to load content
    output_md = "output_classified.md"
    output_csv = "output_classified.csv"
    generate_reports(final_classifications, st.session_state.questions, output_md, output_csv)
    
    # Load formatted markdown content
    with open(output_md, "r", encoding="utf-8") as f:
        md_content = f.read()
        
    # Generate text format content
    text_lines = ["--- Question Mappings ---"]
    for c in final_classifications:
        text_lines.append(f"Q.{c['id']} -> {c['category']}")
    text_content = "\n".join(text_lines)
    
    with tab_markdown:
        st.subheader("Markdown Document Preview")
        st.markdown("```markdown\n" + md_content + "\n```")
        
    with tab_text:
        st.subheader("Raw Mappings Text Preview")
        st.code(text_content, language="text")
        
    # Export bar (download buttons)
    st.subheader("📥 Export Final Outputs")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label="📥 Download Markdown (.md)",
            data=md_content,
            file_name="output_classified.md",
            mime="text/markdown"
        )
    with col2:
        with open(output_csv, "r", encoding="utf-8") as f:
            csv_content = f.read()
        st.download_button(
            label="📥 Download CSV (.csv)",
            data=csv_content,
            file_name="output_classified.csv",
            mime="text/csv"
        )
    with col3:
        st.download_button(
            label="📥 Download Raw Text (.txt)",
            data=text_content,
            file_name="output_classified.txt",
            mime="text/plain"
        )
