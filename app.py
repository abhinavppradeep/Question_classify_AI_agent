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
    page_title="Gemini Question Classifier Suite",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Premium Dark/Glassmorphism theme
st.markdown("""
<style>
    .reportview-container {
        background: #0f1116;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #e2e8f0 !important;
        font-family: 'Outfit', sans-serif;
    }
    .stButton>button {
        background-color: #4f46e5;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #4338ca;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: #94a3b8;
        font-size: 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #e2e8f0 !important;
        border-bottom: 2px solid #4f46e5 !important;
    }
</style>
""", unsafe_allow_html=True)

# App Title
st.title("🔮 Gemini Question Classifier Suite")
st.caption("A fully-fledged extraction and classification workspace leveraging multi-model fallbacks and live output verification.")

# Initialize Session State
if "questions" not in st.session_state:
    st.session_state.questions = []
if "classifications" not in st.session_state:
    st.session_state.classifications = []
if "categories" not in st.session_state:
    st.session_state.categories = []

# Sidebar Config
st.sidebar.header("⚙️ Configuration")

# API Key Config
default_api_key = os.environ.get("GEMINI_API_KEY", "")
api_key = st.sidebar.text_input("Gemini API Key", value=default_api_key, type="password")
if api_key != default_api_key:
    os.environ["GEMINI_API_KEY"] = api_key
    # Persist in .env
    set_key(".env", "GEMINI_API_KEY", api_key)

# Model Selection Dropdowns
st.sidebar.subheader("🤖 LLM Model Selection")
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

# File Uploader
uploaded_file = st.file_uploader("Upload PDF Exam Paper (or cached OCR text file)", type=["pdf", "txt"])

if uploaded_file is not None:
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.success(f"Uploaded: {uploaded_file.name}")
    
    # Process Button
    if st.button("🚀 Start Extraction & Classification"):
        # Clear previous state to prevent stale data display during execution
        st.session_state.questions = []
        st.session_state.classifications = []
        
        from src.logger import setup_logger
        run_logger = setup_logger("run.log")
        
        try:
            with st.status("Processing PDF...", expanded=True) as status:
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
    st.header("🔍 Workspace Verification & Outputs")
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
