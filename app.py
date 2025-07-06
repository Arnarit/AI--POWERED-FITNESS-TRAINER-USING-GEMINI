import streamlit as st
import os
from PIL import Image
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
# load_dotenv is no longer needed
import tempfile
import time
import io
import shutil # For robust directory cleanup

# --- Environment Setup is now handled by user input ---

# --- Constants ---
FAISS_INDEX_PATH = "faiss_index"
# List explicitly supported video MIME types by Gemini API (refer to documentation for updates)
SUPPORTED_VIDEO_TYPES = ["mp4", "mpeg", "mov", "avi", "x-flv", "x-ms-wmv", "webm", "quicktime", "mpg", "wmv", "flv"]
SUPPORTED_VIDEO_MIMETYPES = [f"video/{ext}" for ext in SUPPORTED_VIDEO_TYPES]

# --- CSS Styling (remains identical to your original code) ---
st.markdown("""
<style>
/* ... [Your existing CSS remains unchanged] ... */
body {
    color: #E0E0E0; /* Lighter grey for better readability */
    background-color: #0a0a0f; /* Slightly darker, richer background */
    font-family: 'Roboto', sans-serif; /* Modern, readable font */
}
.stApp {
    background: linear-gradient(145deg, #111827, #0a0a0f 80%); /* Subtle dark gradient */
}
h1, h2, h3, h4, h5, h6 {
    color: #2ECC71;  /* Vibrant but slightly softer green */
    font-weight: 600;
    text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
}
p, .stMarkdown {
    color: #C5C5C5; /* Light grey for body text */
    line-height: 1.6;
}
hr {
    border-top: 1px solid #2ECC71;
    opacity: 0.3;
}
.stSidebar > div:first-child {
    background-color: rgba(17, 24, 39, 0.85); /* Darker, semi-transparent sidebar */
    backdrop-filter: blur(8px);
    border-right: 1px solid rgba(46, 204, 113, 0.3);
    padding-top: 1rem;
}
.stSidebar [data-testid="stRadio"] > label,
.stSidebar .stButton>button {
    color: #E0E0E0; /* White text for sidebar elements */
}
.stSidebar [data-testid="stRadio"] {
    margin-bottom: 1rem;
    padding: 0.5rem;
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
}
.stSidebar .stButton>button {
    background-color: #2ECC71;
    color: #0a0a0f; /* Dark text on green button */
    border-radius: 20px;
    border: none;
    padding: 10px 20px;
    font-size: 1em;
    font-weight: 600; /* Bolder button text */
    box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.4);
    transition: all 0.3s ease;
    width: 100%;
    margin-top: 0.5rem; /* Space between buttons */
}
.stSidebar .stButton>button:hover {
    background-color: #58D68D; /* Lighter green on hover */
    transform: translateY(-2px);
    box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.5);
}
.stSidebar .stButton>button:active {
    transform: translateY(0px);
    box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.3);
}
.stSidebar button[kind="secondary"] { /* More specific selector for secondary button */
    background-color: #e74c3c; /* Red for clear/destructive action */
    color: #ffffff;
}
.stSidebar button[kind="secondary"]:hover {
     background-color: #f1948a; /* Lighter red on hover */
}
.stChatInputContainer { /* Target the container div of chat input */
   background-color: #0a0a0f; /* Match app background */
   border-top: 1px solid rgba(46, 204, 113, 0.3); /* Subtle top border */
   padding: 0.75rem 1rem; /* Adjust padding */
}
[data-testid="stChatInput"] textarea {
    background-color: rgba(255, 255, 255, 0.05); /* Very subtle background */
    color: #E0E0E0;
    border: 1px solid rgba(46, 204, 113, 0.5); /* Green border */
    border-radius: 8px;
    transition: border-color 0.3s ease;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #2ECC71; /* Brighter green on focus */
    box-shadow: 0 0 5px rgba(46, 204, 113, 0.5);
}
[data-testid="stChatMessage"] {
    padding: 0.75rem 1rem;
    border-radius: 12px;
    margin-bottom: 0.75rem;
    max-width: 80%; /* Limit width slightly more */
    word-wrap: break-word;
    box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.2);
}
[data-testid="stChatMessageContent"] { /* Target inner content for padding */
     padding: 0; /* Reset inner padding if needed */
}
div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) {
    background-color: rgba(46, 204, 113, 0.2); /* Lighter green background */
    align-self: flex-end;
    margin-left: auto; /* Push to right */
    border-radius: 12px 12px 3px 12px;
}
div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) {
    background-color: rgba(50, 50, 60, 0.9); /* Slightly lighter dark */
    align-self: flex-start;
    margin-right: auto; /* Push to left */
    border-radius: 12px 12px 12px 3px;
}
.stFileUploader {
    background-color: rgba(255, 255, 255, 0.03);
    border: 2px dashed #2ECC71;
    border-radius: 10px;
    padding: 15px; /* Slightly reduced padding */
    margin-bottom: 1rem; /* Add space below uploader */
    text-align: center;
}
.stFileUploader label {
    color: #2ECC71; /* Green text for label */
    font-weight: 500;
}
.stFileUploader [data-testid="stFileUploaderFile"] {
    color: #E0E0E0;
}
.stFileUploader [data-testid="stFileUploaderFileName"] {
    font-weight: bold;
}
.stProgress > div > div > div > div {
    background-color: #2ECC71; /* Green progress bar */
}
.stSpinner > div {
    border-top-color: #2ECC71; /* Spinner color */
}
.stAlert { /* General styling for alerts */
     border-radius: 8px;
     border: 1px solid;
     padding: 1rem;
     font-weight: 500;
}
.stAlert[data-baseweb="notification-positive"] { /* Success */
    background-color: rgba(46, 204, 113, 0.1);
    color: #2ECC71;
    border-color: #2ECC71;
}
.stAlert[data-baseweb="notification-negative"] { /* Error */
     background-color: rgba(231, 76, 60, 0.1);
     color: #e74c3c;
     border-color: #e74c3c;
}
.stAlert[data-baseweb="notification-warning"] { /* Warning */
     background-color: rgba(243, 156, 18, 0.1);
     color: #f39c12;
     border-color: #f39c12;
}
.stImage > div { /* Target container for image display */
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);
    margin: 10px auto 20px auto; /* Center image and add bottom margin */
    max-width: 90%; /* Ensure image respects container boundaries */
}
.stImage img {
    display: block;
    width: 100%; /* Make image responsive */
    height: auto;
}
[data-testid="stHorizontalBlock"] > div {
    padding: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# --- Session State Initialization ---
# NEW: Add state for API key management
if "api_key_configured" not in st.session_state:
    st.session_state.api_key_configured = False
if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = None

# Your original session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Namaste! 🙏 Please enter your Google API Key to activate the trainer."}]
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "💬 General Chat & Image"
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False
if "uploaded_video_uri" not in st.session_state:
    st.session_state.uploaded_video_uri = None
if "uploaded_video_temp_dir" not in st.session_state:
    st.session_state.uploaded_video_temp_dir = None
if "chat_image_uploader_key" not in st.session_state:
    st.session_state.chat_image_uploader_key = 0
if "current_chat_image" not in st.session_state:
    st.session_state.current_chat_image = None
if "current_chat_image_parts" not in st.session_state:
     st.session_state.current_chat_image_parts = None


# --- Helper, PDF, and API Functions (remains identical to your original code) ---
def safe_cleanup_dir(dir_path):
    if dir_path and os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            return True
        except OSError as e:
            st.warning(f"⚠️ Could not completely clean up temporary directory {os.path.basename(dir_path)}: {e}")
            return False
    return True

def reset_chat():
    st.session_state.messages = [{"role": "assistant", "content": "Chat cleared! How can I help you next?"}]
    st.session_state.pdf_processed = False
    st.session_state.video_processed = False
    st.session_state.uploaded_video_uri = None
    if "uploaded_video_temp_dir" in st.session_state:
        safe_cleanup_dir(st.session_state.uploaded_video_temp_dir)
        st.session_state.uploaded_video_temp_dir = None
    st.session_state.current_chat_image = None
    st.session_state.current_chat_image_parts = None
    st.session_state.chat_image_uploader_key += 1
    if os.path.exists(FAISS_INDEX_PATH):
        safe_cleanup_dir(FAISS_INDEX_PATH)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            if not pdf_reader.pages:
                 st.warning(f"⚠️ PDF '{pdf.name}' contains no pages or could not be read.")
                 continue
            for i, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as page_e:
                     st.warning(f"⚠️ Error extracting text from page {i+1} of '{pdf.name}': {page_e}")
        except Exception as e:
            st.warning(f"⚠️ Could not read PDF '{pdf.name}': {e}. Skipping.")
    return text

def get_text_chunks(text):
    if not text or not text.strip():
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=1000,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, embeddings_model):
    if not text_chunks:
        st.warning("⚠️ No text chunks available to create vector store.")
        return False
    try:
        if os.path.exists(FAISS_INDEX_PATH):
             safe_cleanup_dir(FAISS_INDEX_PATH)
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings_model)
        vector_store.save_local(FAISS_INDEX_PATH)
        st.success(f"✅ FAISS index created with {len(text_chunks)} chunks.")
        return True
    except Exception as e:
        st.error(f"🔴 Error creating/saving vector store: {e}")
        safe_cleanup_dir(FAISS_INDEX_PATH)
        return False

def get_conversational_chain(langchain_chat_model):
    prompt_template = """... [Your original prompt template remains unchanged] ..."""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    try:
        chain = load_qa_chain(langchain_chat_model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"🔴 Error loading QA chain: {e}")
        return None

def handle_pdf_query(user_question, embeddings_model, langchain_chat_model):
    if not os.path.exists(FAISS_INDEX_PATH) or not os.listdir(FAISS_INDEX_PATH):
        st.error("🔴 FAISS index not found or is empty. Please process PDF documents first.")
        return "Error: Document index not available."
    if not user_question:
        st.warning("⚠️ Please enter a question about the processed PDFs.")
        return None
    try:
        st.info("🔍 Searching PDF index...")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question, k=5)
        if not docs:
            return "I couldn't find information relevant to your question in the processed documents."
        chain = get_conversational_chain(langchain_chat_model)
        if not chain:
             return "Error: Could not initialize the analysis chain."
        st.info("🧠 Analyzing relevant document sections...")
        response = chain.invoke({"input_documents": docs, "question": user_question})
        output_text = response.get("output_text", "Error: Could not extract response text.")
        return output_text
    except Exception as e:
        st.error(f"🔴 Error during PDF query processing: {e}")
        return "An error occurred while searching the documents."

def get_gemini_response_text_image(model, system_prompt, text_input=None, image_parts=None):
    # This function is identical to your original
    content_request = []
    current_input = ""
    if text_input:
        current_input = f"{system_prompt}\n\n--- User Query ---\n{text_input}"
    elif image_parts:
        current_input = f"{system_prompt}\n\n--- Task ---\nAnalyze the provided image based on the instructions above."
    else:
        return "Please provide text input or upload an image to get a response."
    if image_parts:
        content_request.extend(image_parts)
    content_request.append(current_input)
    try:
        response = model.generate_content(content_request, request_options={"timeout": 300})
        return response.text
    except Exception as e:
        st.error(f"🔴 Error communicating with Gemini API: {e}")
        return "Sorry, I encountered an error trying to generate a response."

def get_gemini_response_video(model, prompt_instructions, video_uri, user_question):
    # This function is identical to your original
    if not video_uri or not os.path.exists(video_uri):
        return "❌ Error: Video file path is invalid or file does not exist. Please re-upload."
    video_file_api = None
    progress_bar = st.progress(0, text="Initiating video upload...")
    analysis_placeholder = st.empty()
    try:
        analysis_placeholder.info(f"⏳ Uploading '{os.path.basename(video_uri)}' to Google API...")
        video_file_api = genai.upload_file(path=video_uri)
        analysis_placeholder.info(f"✅ Upload initiated. Waiting for processing...")
        polling_start_time = time.time()
        timeout_seconds = 600
        while time.time() - polling_start_time < timeout_seconds:
            video_file_api = genai.get_file(video_file_api.name)
            if video_file_api.state.name == "ACTIVE":
                analysis_placeholder.success("✅ Video processing complete.")
                break
            elif video_file_api.state.name == "FAILED":
                raise ValueError("Google API failed to process the video.")
            time.sleep(10)
        else:
             raise TimeoutError("Video processing timed out.")
        analysis_placeholder.info("🧠 Generating analysis based on video and question...")
        response = model.generate_content(
            [prompt_instructions, user_question, video_file_api],
            request_options={"timeout": 900}
        )
        return response.text
    except Exception as e:
        st.error(f"🔴 Unexpected Error during video analysis: {e}")
        return f"Sorry, an unexpected error occurred during video analysis. Details: {e}"
    finally:
        if video_file_api and hasattr(video_file_api, 'name'):
            try:
                genai.delete_file(video_file_api.name)
            except Exception as delete_e:
                st.warning(f"⚠️ Could not delete API file artifact '{video_file_api.name}': {delete_e}")
        progress_bar.empty()
        analysis_placeholder.empty()

def input_image_setup(uploaded_file):
    # This function is identical to your original
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [{"mime_type": uploaded_file.type, "data": bytes_data}]
        st.session_state.current_chat_image = uploaded_file
        st.session_state.current_chat_image_parts = image_parts
        img = Image.open(io.BytesIO(bytes_data))
        st.image(img, caption=f"Uploaded: {uploaded_file.name}", use_column_width='auto')
        return image_parts
    st.session_state.current_chat_image = None
    st.session_state.current_chat_image_parts = None
    return None

def input_video_setup(uploaded_file):
    # This function is identical to your original
    if uploaded_file is not None:
        if "uploaded_video_temp_dir" in st.session_state:
             safe_cleanup_dir(st.session_state.uploaded_video_temp_dir)
        temp_dir = tempfile.mkdtemp()
        temp_video_path = os.path.join(temp_dir, f"uploaded_video{os.path.splitext(uploaded_file.name)[1]}")
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ Video '{uploaded_file.name}' saved locally.")
        st.video(temp_video_path)
        st.session_state.video_processed = True
        st.session_state.uploaded_video_uri = temp_video_path
        st.session_state.uploaded_video_temp_dir = temp_dir
        return temp_video_path
    st.session_state.video_processed = False
    st.session_state.uploaded_video_uri = None
    safe_cleanup_dir(st.session_state.get("uploaded_video_temp_dir"))
    st.session_state.uploaded_video_temp_dir = None
    return None

# --- Prompt Templates (remains identical to your original code) ---
TEXT_IMAGE_PROMPT = """... [Your original TEXT_IMAGE_PROMPT remains unchanged] ..."""
VIDEO_ANALYSIS_PROMPT = """... [Your original VIDEO_ANALYSIS_PROMPT remains unchanged] ..."""


# --- Sidebar UI ---
with st.sidebar:
    st.title("AI Fitness Trainer 🧘‍♂️")
    st.markdown("---")

    # NEW: API Key Input Section
    st.header("🔑 API Configuration")
    if not st.session_state.api_key_configured:
        api_key_input = st.text_input(
            "Enter your Google API Key:",
            type="password",
            help="Get your key from Google AI Studio.",
            key="api_key_input_field"
        )
        if st.button("Activate Trainer", key="activate_api"):
            if api_key_input:
                try:
                    # Validate the key by configuring the genai library
                    genai.configure(api_key=api_key_input)
                    # A light test to confirm connectivity
                    genai.list_models()
                    st.session_state.google_api_key = api_key_input
                    st.session_state.api_key_configured = True
                    st.session_state.messages = [{"role": "assistant", "content": "Namaste! 🙏 API Key accepted. How can I help you today?"}]
                    st.success("✅ API Key is valid. The trainer is now active!")
                    time.sleep(1.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"🔴 Invalid API Key. Please check and try again. Error: {e}")
            else:
                st.warning("Please enter your Google API Key.")
    else:
        st.success("API Key is Active")
        if st.button("Change API Key"):
            st.session_state.api_key_configured = False
            st.session_state.google_api_key = None
            st.session_state.messages = [{"role": "assistant", "content": "Namaste! 🙏 Please enter a new Google API Key to activate the trainer."}]
            reset_chat()
            st.rerun()

    st.markdown("---")

    # The rest of the sidebar is only shown if the key is configured
    if st.session_state.api_key_configured:
        # Your original sidebar code
        app_mode = st.radio(
            "Choose Interaction Mode:",
            ("💬 General Chat & Image", "📄 PDF Report Q&A", "🎬 Video Analysis"),
            key="app_mode_radio",
            index=["💬 General Chat & Image", "📄 PDF Report Q&A", "🎬 Video Analysis"].index(st.session_state.app_mode)
        )
        if app_mode != st.session_state.app_mode:
            st.session_state.app_mode = app_mode
            st.rerun()

        st.markdown("---")
        st.sidebar.markdown("### ✨ AI Capabilities")
        st.sidebar.markdown("""
        - 🏋️ Personalized Recommendations
        - 🍎 Nutritional Guidance
        - 🧠 Mental Wellness Support
        - 🔬 PDF Report Analysis (RAG)
        - 🎬 Video Content Analysis
        - 🗣️ Natural Language Chat
        - 👀 Image Recognition
        """)

        if st.session_state.app_mode == "📄 PDF Report Q&A":
            st.markdown("---")
            st.markdown("### 📄 PDF Actions")
            pdf_docs = st.file_uploader("Upload Medical/Fitness Reports (PDF)", type="pdf", accept_multiple_files=True)
            if st.button("Process Uploaded PDFs", key="process_pdfs"):
                if pdf_docs:
                    # Models are initialized in the main app block, so they are available here
                    with st.spinner("⚙️ Processing PDFs..."):
                        raw_text = get_pdf_text(pdf_docs)
                        if raw_text.strip():
                            text_chunks = get_text_chunks(raw_text)
                            if get_vector_store(text_chunks, embeddings):
                                st.session_state.pdf_processed = True
                        else:
                            st.warning("⚠️ No text could be extracted from the uploaded PDF(s).")
                else:
                    st.warning("⚠️ Please upload at least one PDF file.")

        elif st.session_state.app_mode == "🎬 Video Analysis":
            st.markdown("---")
            st.markdown("### 🎬 Video Actions")
            if st.session_state.get("video_processed"):
                st.success(f"Video Ready: {os.path.basename(st.session_state.uploaded_video_uri)}")
                if st.button("Upload Different Video"):
                    input_video_setup(None)
                    st.rerun()
            else:
                 uploaded_video = st.file_uploader("Upload Video for Analysis", type=SUPPORTED_VIDEO_TYPES)
                 if uploaded_video:
                      with st.spinner("⏳ Saving & verifying..."):
                          if input_video_setup(uploaded_video):
                               st.rerun()

        st.markdown("---")
        st.markdown("### ✨ Controls")
        if st.button("Clear Chat History", key="clear_chat", type="secondary"):
            reset_chat()
            st.rerun()

    st.markdown("---")
    st.info("ℹ️ Consult professionals for personalized fitness/medical advice.")

# --- Main Application Logic ---

# If API key is not configured, show a welcome screen and stop.
if not st.session_state.api_key_configured:
    st.header("Welcome to the AI Fitness Trainer!")
    st.info("To get started, please enter your Google API Key in the sidebar.")
    st.markdown("You can get a free API Key from [Google AI Studio](https://aistudio.google.com/app/apikey).")
    with st.chat_message(st.session_state.messages[0]["role"]):
        st.markdown(st.session_state.messages[0]["content"])
    st.stop() # This is crucial to prevent the rest of the app from running

# --- Model Initialization (runs only if key is valid) ---
# This block is inside the main script flow, so it runs after the st.stop() check
try:
    API_KEY = st.session_state.google_api_key
    genai.configure(api_key=API_KEY)
    
    # Using your original model names
    base_model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
    langchain_chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0.7, google_api_key=API_KEY)
    embedding_model_name = "models/text-embedding-004"
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name, google_api_key=API_KEY)

except Exception as e:
    st.error(f"🔴 Error initializing Generative AI models. There might be an issue with the API key or service. Please try changing the key.")
    st.stop()


# --- Main Chat Area (your original code, runs only if key is valid) ---
st.header(f" {st.session_state.app_mode}")
st.markdown("---")

if st.session_state.app_mode == "💬 General Chat & Image":
    uploaded_image_file = st.file_uploader(
        "Upload an image for analysis (Optional)",
        type=["jpeg", "jpg", "png", "webp"],
        key=f"chat_image_uploader_{st.session_state.chat_image_uploader_key}"
    )
    if uploaded_image_file != st.session_state.get("current_chat_image"):
         input_image_setup(uploaded_image_file)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your fitness/health question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = None
        message_placeholder = st.empty()
        message_placeholder.markdown("🧠 Thinking...")

        if st.session_state.app_mode == "💬 General Chat & Image":
            response = get_gemini_response_text_image(
                base_model, TEXT_IMAGE_PROMPT,
                text_input=prompt, image_parts=st.session_state.get("current_chat_image_parts")
            )
        elif st.session_state.app_mode == "📄 PDF Report Q&A":
            if st.session_state.pdf_processed:
                response = handle_pdf_query(prompt, embeddings, langchain_chat_model)
            else:
                response = "⚠️ Please upload and process PDF documents first using the sidebar."
        elif st.session_state.app_mode == "🎬 Video Analysis":
            if st.session_state.get("video_processed"):
                message_placeholder.empty()
                response = get_gemini_response_video(
                    base_model, VIDEO_ANALYSIS_PROMPT,
                    st.session_state.uploaded_video_uri, prompt
                )
            else:
                 response = "⚠️ Please upload a video using the sidebar first."

        if response:
            message_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
             error_message = "Could not generate a response. Please check inputs/API key, or try again."
             message_placeholder.warning(error_message)
             st.session_state.messages.append({"role": "assistant", "content": error_message})

# Contextual Footer Info
if st.session_state.app_mode == "📄 PDF Report Q&A":
    if not st.session_state.pdf_processed:
        st.info("ℹ️ Upload PDF reports via the sidebar & click 'Process'. Then ask questions.")
    else:
        st.success("✅ PDFs processed. Ask specific questions about the reports.")
elif st.session_state.app_mode == "🎬 Video Analysis":
     if not st.session_state.get("video_processed"):
         st.info("ℹ️ Upload a video via the sidebar. Once ready, ask specific questions.")
     else:
         st.success(f"✅ Video '{os.path.basename(st.session_state.uploaded_video_uri)}' is ready. Ask questions about it.")
elif st.session_state.app_mode == "💬 General Chat & Image":
    st.info("ℹ️ Ask general fitness/health questions, or upload an image above for analysis.")
