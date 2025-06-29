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
import tempfile
import time
import io
import shutil # For robust directory cleanup

# --- Constants ---
FAISS_INDEX_PATH = "faiss_index"
# List explicitly supported video MIME types by Gemini API
SUPPORTED_VIDEO_TYPES = ["mp4", "mpeg", "mov", "avi", "x-flv", "x-ms-wmv", "webm", "quicktime", "mpg", "wmv", "flv"]
SUPPORTED_VIDEO_MIMETYPES = [f"video/{ext}" for ext in SUPPORTED_VIDEO_TYPES]


# --- CSS Styling ---
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

/* --- General Elements --- */
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

/* --- Sidebar --- */
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
/* Specific style for Clear Chat button */
.stSidebar button[kind="secondary"] { /* More specific selector for secondary button */
    background-color: #e74c3c; /* Red for clear/destructive action */
    color: #ffffff;
}
.stSidebar button[kind="secondary"]:hover {
     background-color: #f1948a; /* Lighter red on hover */
}


/* --- Chat Interface --- */
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

/* Chat Messages */
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

/* User Message */
div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) {
    background-color: rgba(46, 204, 113, 0.2); /* Lighter green background */
    align-self: flex-end;
    margin-left: auto; /* Push to right */
    border-radius: 12px 12px 3px 12px;
}

/* AI Message */
div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) {
    background-color: rgba(50, 50, 60, 0.9); /* Slightly lighter dark */
    align-self: flex-start;
    margin-right: auto; /* Push to left */
    border-radius: 12px 12px 12px 3px;
}


/* --- File Uploader --- */
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
/* Style for uploaded file display */
.stFileUploader [data-testid="stFileUploaderFile"] {
    color: #E0E0E0;
}
.stFileUploader [data-testid="stFileUploaderFileName"] {
    font-weight: bold;
}


/* --- Other Elements --- */
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
/* Add some spacing for columns */
[data-testid="stHorizontalBlock"] > div {
    padding: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Namaste! 🙏 Please enter your Google API Key to begin."}]
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


# --- Helper Functions ---

def safe_cleanup_dir(dir_path):
    """Safely removes a directory if it exists."""
    if dir_path and os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            return True
        except OSError as e:
            st.warning(f"⚠️ Could not completely clean up temporary directory {os.path.basename(dir_path)}: {e}")
            return False
    return True

def reset_chat():
    """Clears chat history and resets related states, including temp files."""
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

def get_vector_store(text_chunks):
    if not text_chunks:
        st.warning("⚠️ No text chunks available to create vector store.")
        return False
    try:
        if os.path.exists(FAISS_INDEX_PATH):
             safe_cleanup_dir(FAISS_INDEX_PATH)
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        st.success(f"✅ FAISS index created with {len(text_chunks)} chunks.")
        return True
    except Exception as e:
        st.error(f"🔴 Error creating/saving vector store: {e}")
        safe_cleanup_dir(FAISS_INDEX_PATH)
        return False

def get_conversational_chain():
    prompt_template = """
        As an fitness trainers you are expert in understanding healthcare domain including medical, medicines, fitness, sports and Brahmacharya.
    You have extreme deep knowledge of medical sciences, Brahmacharya,yoga and exercises, physiological and psychological fitness, spiritual and ayurveda.
    We will ask you many questions on health domain and you will have to answer any questions.
    Answer the question as detailed as possible from the provided context, make sure to provide all the details in any languages based on user input:
    Disclaimer: This AI Fitness Trainer application provides information and recommendations for general fitness and wellness purposes only.
    It is not intended legally,and should not be used as, a substitute for professional medical advice, diagnosis, or treatment.
    Always seek the advice of your physician or other qualified health provider physically with any questions you may have regarding
    a medical condition for getting better benefits.
    Never disregard professional medical advice or delay in seeking it because of something you have read or received through this application.
    1. Provide all the information about the health problem and types of sides effects and how to recover from it based on context
    2. types of exercises and yoga fitness for different person based on context to recover from health problem and become fit
    3. recommendations of diet plans and types of foods according to clock timing for different weather conditions
       for recovering from health problem and becoming fit based on context
    4. Mental fitness exercise for stress, anxiety, depression and other mental health issues based on provided context
    5. recommendation of fitness plans and lifestyles according to clock timing for different weather conditions
    6. ayurvedic and natural remedies for health problem based on context
    7. will he/she involved in sports? if yes, which sports based on problems given context?
    8. What should he/she avoid to recover from problem based on context?
    9. Disclaimer: This AI Fitness Trainer application provides information and recommendations for general fitness and wellness purposes only.
       It is not intended legally,and should not be used as, a substitute for professional medical advice, diagnosis, or treatment.
       Always seek the advice of your physician or other qualified health provider physically with any questions you may have regarding
       a medical condition for getting better benefits.
       Never disregard professional medical advice or delay in seeking it because of something you have read or received through this application.
       Now general suggestion on types of medicines , supplements and medical treaments that can be used to recover from health problem
       if it is required otherwise recommend to the doctor

    If uploded pdf {context} is not related to health , fitness report, medical test report , medicines domain :
    1.  then say just this line : "Sorry, I am an AI fitness trainer, I can only answer questions
        related to health domain. Please ask a question related to health domain.", don't provide the wrong answer.
    2.  If question is on summarisation of {context} which is not related to health domain, medical test report , sports, fitness test report ,yoga or exercises and medicines then
        say just this line : "Sorry, I am an AI fitness trainer, I can only answer questions
        related to health domain. Please ask a question related to health domain.", don't provide the wrong answer.


    If question is not related to health domain, medical , sports, fitness,yoga or exercises and medicines then say just this line : "Sorry, I am an AI fitness trainer, I can only answer questions
    related to health domain. Please ask a question related to health domain.", don't provide the wrong answer.

    if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer should be in between 1000 to 100000 numbers of words:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    try:
        chain = load_qa_chain(langchain_chat_model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"🔴 Error loading QA chain: {e}")
        return None

def handle_pdf_query(user_question):
    if not os.path.exists(FAISS_INDEX_PATH) or not os.listdir(FAISS_INDEX_PATH):
        st.error("🔴 FAISS index not found or is empty. Please process PDF documents first using the sidebar.")
        return "Error: Document index not available. Please process PDFs via the sidebar."
    if not user_question:
        st.warning("⚠️ Please enter a question about the processed PDFs.")
        return None
    try:
        st.info("🔍 Searching PDF index...")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question, k=5)
        if not docs:
            return "I couldn't find information relevant to your question in the processed documents."
        chain = get_conversational_chain()
        if not chain:
             return "Error: Could not initialize the analysis chain."
        st.info("🧠 Analyzing relevant document sections...")
        response = chain.invoke({"input_documents": docs, "question": user_question})
        output_text = response.get("output_text", "Error: Could not extract response text.")
        if "answer is not available in the provided document context" in output_text.lower() and len(output_text) < 100:
             return "The answer to your question is not available in the provided document context."
        return output_text
    except Exception as e:
        st.error(f"🔴 Error during PDF query processing: {e}")
        if "Index doesn't look like a FAISS index" in str(e) or "FileNotFoundError" in str(e):
             st.error("🔴 The FAISS index file might be corrupted or missing. Please try processing the PDFs again.")
        return "An error occurred while searching the documents."

def get_gemini_response_text_image(system_prompt: str, text_input: str = None, image_parts: list = None):
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
    if not content_request:
         return "Internal error: No content generated for API request."
    try:
        response = base_model.generate_content(content_request, request_options={"timeout": 300})
        return response.text
    except Exception as e:
        st.error(f"🔴 Error communicating with Gemini API: {e}")
        if "API key not valid" in str(e):
            st.error("🔴 API Key Error: Please check if your GOOGLE_API_KEY is correct and enabled.")
        elif "quota" in str(e).lower():
             st.error("🔴 Quota Error: You might have exceeded your API request quota.")
        elif "Deadline Exceeded" in str(e):
             st.error("🔴 Timeout Error: The request took too long to process.")
        elif "Invalid content" in str(e):
             st.error(f"🔴 Content Error: The API rejected the input format or content. Details: {e}")
        else:
             st.error(f"🔴 Unknown API Error: {e}")
        return "Sorry, I encountered an error trying to generate a response."

def get_gemini_response_video(prompt_instructions: str, video_uri: str, user_question: str):
    if not video_uri or not os.path.exists(video_uri):
        return "❌ Error: Video file path is invalid or file does not exist. Please re-upload."
    if not user_question:
        return "⚠️ Please ask a specific question about the video."
    video_file_api = None
    progress_bar = st.progress(0, text="Initiating video upload...")
    analysis_placeholder = st.empty()
    try:
        analysis_placeholder.info(f"⏳ Uploading '{os.path.basename(video_uri)}' to Google API...")
        video_file_api = genai.upload_file(path=video_uri)
        progress_bar.progress(20, text="Upload initiated. Waiting for API processing...")
        analysis_placeholder.info(f"✅ Upload initiated. File Name: {video_file_api.name}. Waiting for processing...")
        polling_start_time = time.time()
        timeout_seconds = 600
        poll_interval = 15
        max_retries = 3
        retries = 0
        while True:
            current_time = time.time()
            if current_time - polling_start_time > timeout_seconds:
                raise TimeoutError(f"Video processing timed out after {timeout_seconds} seconds.")
            try:
                video_file_api = genai.get_file(video_file_api.name)
                file_state = video_file_api.state.name
                retries = 0
                if file_state == "ACTIVE":
                    progress_bar.progress(60, text="Video processed by API. Ready for analysis.")
                    analysis_placeholder.success("✅ Video processing complete.")
                    break
                elif file_state == "FAILED":
                    raise ValueError(f"Google API failed to process the video. State: {file_state}. Reason: {getattr(video_file_api, 'error', 'Unknown')}")
                elif file_state == "PROCESSING":
                    elapsed_time = int(current_time - polling_start_time)
                    progress = 20 + int(40 * elapsed_time / timeout_seconds)
                    progress_bar.progress(min(progress, 59), text=f"API processing video... ({elapsed_time}s elapsed)")
                    analysis_placeholder.info(f"⏳ API processing video... State: {file_state} ({elapsed_time}s elapsed)")
                else:
                    analysis_placeholder.warning(f"⏳ Video in unexpected state: {file_state}. Continuing to poll...")
            except Exception as poll_e:
                retries += 1
                st.warning(f"⚠️ Error polling video status ({retries}/{max_retries}): {poll_e}. Retrying in {poll_interval}s...")
                if retries >= max_retries:
                    raise ConnectionError(f"Failed to get video status after {max_retries} retries: {poll_e}") from poll_e
                time.sleep(poll_interval)
                continue
            time.sleep(poll_interval)
        progress_bar.progress(70, text="Sending analysis request to Gemini...")
        analysis_placeholder.info("🧠 Generating analysis based on video and question...")
        full_prompt_content = [prompt_instructions, user_question, video_file_api]
        response = base_model.generate_content(full_prompt_content, request_options={"timeout": 900})
        progress_bar.progress(100, text="Analysis complete!")
        analysis_placeholder.success("✅ Analysis received from Gemini.")
        time.sleep(1)
        return response.text
    except TimeoutError as e:
        st.error(f"🔴 Timeout Error: {e}")
        return f"Error: Video processing took too long ({timeout_seconds}s limit). Please try a shorter video or try again later."
    except ValueError as e:
        st.error(f"🔴 Processing Error: {e}")
        return f"Error: The video could not be processed by the API. It might be corrupted or in an unsupported format."
    except ConnectionError as e:
         st.error(f"🔴 Connection Error: {e}")
         return "Error: Could not reliably check video processing status. Please check your connection and try again."
    except Exception as e:
        st.error(f"🔴 Unexpected Error during video analysis: {e}")
        if "API key not valid" in str(e):
            st.error("🔴 API Key Error: Please check if your GOOGLE_API_KEY is correct and enabled.")
        elif "quota" in str(e).lower():
             st.error("🔴 Quota Error: You might have exceeded your API request quota for generation or file storage.")
        elif "Deadline Exceeded" in str(e):
             st.error("🔴 Timeout Error: The analysis request itself timed out.")
        elif "Invalid content" in str(e) or "Unsupported format" in str(e):
             st.error(f"🔴 Content/Format Error: The API rejected the video or request format. Details: {e}")
        return f"Sorry, an unexpected error occurred during video analysis. Details: {e}"
    finally:
        if video_file_api and hasattr(video_file_api, 'name'):
            try:
                analysis_placeholder.info(f"🧹 Cleaning up API file artifact: {video_file_api.name}...")
                genai.delete_file(video_file_api.name)
                analysis_placeholder.info(f"✅ API file artifact deleted.")
                time.sleep(1)
            except Exception as delete_e:
                st.warning(f"⚠️ Could not delete API file artifact '{video_file_api.name}': {delete_e}")
        progress_bar.empty()
        analysis_placeholder.empty()

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        try:
            bytes_data = uploaded_file.getvalue()
            image_parts = [{"mime_type": uploaded_file.type, "data": bytes_data}]
            st.session_state.current_chat_image = uploaded_file
            st.session_state.current_chat_image_parts = image_parts
            img = Image.open(io.BytesIO(bytes_data))
            with st.container():
                 st.image(img, caption=f"Uploaded: {uploaded_file.name}", use_column_width='auto')
            return image_parts
        except Exception as e:
            st.error(f"🔴 Error processing image '{uploaded_file.name}': {e}")
            st.session_state.current_chat_image = None
            st.session_state.current_chat_image_parts = None
            return None
    else:
        if st.session_state.current_chat_image:
             st.session_state.current_chat_image = None
             st.session_state.current_chat_image_parts = None
        return None

def input_video_setup(uploaded_file):
    if uploaded_file is not None:
        if "uploaded_video_temp_dir" in st.session_state:
             safe_cleanup_dir(st.session_state.uploaded_video_temp_dir)
             st.session_state.uploaded_video_temp_dir = None
             st.session_state.uploaded_video_uri = None
             st.session_state.video_processed = False
        temp_dir = None
        try:
            if uploaded_file.type not in SUPPORTED_VIDEO_MIMETYPES:
                 st.error(f"🔴 Unsupported video format: '{uploaded_file.type}'. Please upload one of: {', '.join(SUPPORTED_VIDEO_TYPES)}")
                 return None
            temp_dir = tempfile.mkdtemp()
            file_extension = os.path.splitext(uploaded_file.name)[1]
            temp_video_path = os.path.join(temp_dir, f"uploaded_video{file_extension}")
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_size_mb = os.path.getsize(temp_video_path) / (1024 * 1024)
            max_size_mb = 1000
            if file_size_mb > max_size_mb:
                 st.error(f"🔴 Video file is too large ({file_size_mb:.1f} MB). Maximum allowed size is {max_size_mb} MB.")
                 safe_cleanup_dir(temp_dir)
                 return None
            st.success(f"✅ Video '{uploaded_file.name}' ({file_size_mb:.1f} MB) saved locally.")
            st.video(temp_video_path)
            st.session_state.video_processed = True
            st.session_state.uploaded_video_uri = temp_video_path
            st.session_state.uploaded_video_temp_dir = temp_dir
            return temp_video_path
        except Exception as e:
            st.error(f"🔴 Error saving/processing video file: {e}")
            st.session_state.video_processed = False
            st.session_state.uploaded_video_uri = None
            safe_cleanup_dir(temp_dir)
            st.session_state.uploaded_video_temp_dir = None
            return None
    else:
        st.session_state.video_processed = False
        st.session_state.uploaded_video_uri = None
        safe_cleanup_dir(st.session_state.get("uploaded_video_temp_dir"))
        st.session_state.uploaded_video_temp_dir = None
        return None

# --- Prompt Templates ---
TEXT_IMAGE_PROMPT = """
**User Query:**
    As an fitness trainers you are expert in understanding healthcare domain including medical, medicines, fitness, sports and Brahmacharya.
    You have extreme deep knowledge of medical sciences, Brahmacharya,yoga and exercises, physiological and psychological fitness, spiritual and ayurveda.
    We will ask you many questions on health domain and you will have to answer any questions
    of user in medical, scientific and evidence-based manner in between 1000 to 100000 numbers of words in any languages based on user input.

    Disclaimer: This AI Fitness Trainer application provides information and recommendations for general fitness and wellness purposes only.
    It is not intended legally,and should not be used as, a substitute for professional medical advice, diagnosis, or treatment.
    Always seek the advice of your physician or other qualified health provider physically with any questions you may have regarding
    a medical condition for getting better benefits.
    Never disregard professional medical advice or delay in seeking it because of something you have read or received through this application.


    If question is not related to health domain, medical , sports, fitness,yoga or exercises and medicines then say just this line : "Sorry, I am an AI fitness trainer, I can only answer questions
    related to health domain. Please ask a question related to health domain.", don't provide the wrong answer.

    **(Image Analysis if provided)**:
    
    As an fitness trainers you are expert in understanding healthcare domain including medical, medicines, fitness, sports and Brahmacharya.
    You have extreme deep knowledge of medical sciences, Brahmacharya,yoga and exercises, physiological and psychological fitness, spiritual and ayurveda.
    We will ask you many questions on health domain and you will have to answer any questions based on the any types of
    resolutions of uploaded image in medical, scientific and evidence-based in between 1000 to 100000 numbers of words in any languages based on user input.

    Disclaimer: This AI Fitness Trainer application provides information and recommendations for general fitness and wellness purposes only.
    It is not intended legally,and should not be used as, a substitute for professional medical advice, diagnosis, or treatment.
    Always seek the advice of your physician or other qualified health provider physically with any questions you may have regarding
    a medical condition for getting better benefits.
    Never disregard professional medical advice or delay in seeking it because of something you have read or received through this application.


    If question is not related to health domain, medical , sports, fitness,yoga or exercises and medicines then say just this line : "Sorry, I am an AI fitness trainer, I can only answer questions
    related to health domain. Please ask a question related to health domain.", don't provide the wrong answer.

    if the uploaded images is not related to health domain, medical , sports, fitness,yoga or exercises and medicines then just say, "answer is not available in the uploaded images", don't provide the wrong answer

    **Your Detailed Response:**
"""
VIDEO_ANALYSIS_PROMPT = """
    Analyze the uploaded video for content and context. As an fitness trainers you are expert in understanding healthcare domain including medical, medicines, fitness, sports and Brahmacharya.
    You have extreme deep knowledge of medical sciences, Brahmacharya,yoga and exercises, physiological and psychological fitness, spiritual and ayurveda.
    We will ask you many questions on health domain and you will have to answer any questions based on the any types of
    resolutions of uploaded video in medical, scientific and evidence-based in between 1000 to 100000 numbers of words in any languages based on user input .

    Disclaimer: This AI Fitness Trainer application provides information and recommendations for general fitness and wellness purposes only.
    It is not intended legally,and should not be used as, a substitute for professional medical advice, diagnosis, or treatment.
    Always seek the advice of your physician or other qualified health provider physically with any questions you may have regarding
    a medical condition for getting better benefits.
    Never disregard professional medical advice or delay in seeking it because of something you have read or received through this application.


    If question is not related to health domain, medical , sports, fitness,yoga or exercises and medicines then say just this line : "Sorry, I am an AI fitness trainer, I can only answer questions
    related to health domain. Please ask a question related to health domain.", don't provide the wrong answer.

    if uploaded videos is not related to health domain, medical , sports, fitness,yoga or exercises and medicines :
    1.  then just say, "Sorry, I am an AI fitness trainer, I can only answer questions
        related to health domain. Please ask a question related to health domain.", don't provide the wrong answer.
    2.  If question is on summarisation of upoaded video which is not related to health , fitness ,sports, yoga and exercises , medical  , medicines domain
        then say just this line : "Sorry, I am an AI fitness trainer, I can only answer questions
        related to health domain. Please ask a question related to health domain.", don't provide the wrong answer.
                        
    Provide a detailed, user-friendly, and actionable response.
    Answers should be written in between 1000 to 100000 numbers of words in any languages based on user input .

    **Your Detailed Video-Based Analysis and Response:**
"""

# --- Streamlit UI Configuration ---

# --- Sidebar ---
with st.sidebar:
    st.title("AI Fitness Trainer 🧘‍♂️")
    st.markdown("---")

    # --- API Key Input ---
    st.markdown("### 🔑 API Configuration")
    
    # The input field's value is now controlled ONLY by the session state.
    # It will be empty on the first run of a new session.
    api_key_input = st.text_input(
        "Enter your Google API Key",
        type="password",
        value=st.session_state.get("GOOGLE_API_KEY", ""),
        help="Get your key from https://aistudio.google.com/app/apikey"
    )

    # If the user enters a new key, update the session state
    if api_key_input and api_key_input != st.session_state.get("GOOGLE_API_KEY"):
        st.session_state.GOOGLE_API_KEY = api_key_input
        # Rerun to apply the new key immediately and trigger re-initialization
        st.rerun()
    st.markdown("---")

    # Mode Selection
    app_mode = st.radio(
        "Choose Interaction Mode:",
        ("💬 General Chat & Image", "📄 PDF Report Q&A", "🎬 Video Analysis"),
        key="app_mode_radio",
        index=["💬 General Chat & Image", "📄 PDF Report Q&A", "🎬 Video Analysis"].index(st.session_state.app_mode)
    )
    if app_mode != st.session_state.app_mode:
        previous_mode = st.session_state.app_mode
        st.session_state.app_mode = app_mode
        st.session_state.current_chat_image = None
        st.session_state.current_chat_image_parts = None
        st.session_state.chat_image_uploader_key += 1
        if previous_mode == "🎬 Video Analysis":
             safe_cleanup_dir(st.session_state.get("uploaded_video_temp_dir"))
             st.session_state.uploaded_video_uri = None
             st.session_state.uploaded_video_temp_dir = None
             st.session_state.video_processed = False
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
        pdf_docs = st.file_uploader(
            "Upload Medical/Fitness Reports (PDF)", type="pdf", accept_multiple_files=True, key="pdf_uploader"
        )
        if st.button("Process Uploaded PDFs", key="process_pdfs"):
            if pdf_docs:
                with st.spinner("⚙️ Processing PDFs... Extracting text & building index..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text and raw_text.strip():
                        text_chunks = get_text_chunks(raw_text)
                        if text_chunks:
                           if get_vector_store(text_chunks):
                               st.session_state.pdf_processed = True
                           else:
                               st.session_state.pdf_processed = False
                        else:
                            st.warning("⚠️ No text chunks generated (PDF might be empty or text too short).")
                            st.session_state.pdf_processed = False
                    else:
                        st.warning("⚠️ No text could be extracted from the uploaded PDF(s). They might be image-based or corrupted.")
                        st.session_state.pdf_processed = False
            else:
                st.warning("⚠️ Please upload at least one PDF file.")
                st.session_state.pdf_processed = False

    elif st.session_state.app_mode == "🎬 Video Analysis":
        st.markdown("---")
        st.markdown("### 🎬 Video Actions")
        video_ready = (st.session_state.get("video_processed") and
                       st.session_state.get("uploaded_video_uri") and
                       os.path.exists(st.session_state.uploaded_video_uri))
        if video_ready:
            st.success(f"Video Ready: {os.path.basename(st.session_state.uploaded_video_uri)}")
            st.video(st.session_state.uploaded_video_uri)
            if st.button("Upload Different Video", key="upload_new_video"):
                 safe_cleanup_dir(st.session_state.get("uploaded_video_temp_dir"))
                 st.session_state.video_processed = False
                 st.session_state.uploaded_video_uri = None
                 st.session_state.uploaded_video_temp_dir = None
                 st.rerun()
        else:
             uploaded_video = st.file_uploader(
                 "Upload Video for Analysis", type=SUPPORTED_VIDEO_TYPES, key="video_uploader",
                 help=f"Supported formats: {', '.join(SUPPORTED_VIDEO_TYPES)}. Max size recommended: ~1GB"
             )
             if uploaded_video is not None:
                  with st.spinner(f"⏳ Saving & verifying '{uploaded_video.name}'..."):
                      if input_video_setup(uploaded_video):
                           st.rerun()

    st.markdown("---")
    st.markdown("### ✨ Controls")
    if st.button("Clear Chat History", key="clear_chat", type="secondary"):
        reset_chat()
        st.rerun()

    st.markdown("---")
    st.info("ℹ️ Consult professionals for personalized fitness/medical advice.")


# --- API Key Check and Model Initialization ---
# This acts as a gatekeeper. The app will not run until a valid API key is provided.
if not st.session_state.get("GOOGLE_API_KEY"):
    st.error("🔴 Please enter your Google API Key in the sidebar to start.")
    st.info("You can obtain a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).")
    st.stop()

try:
    # Use the key from session state to configure the AI models
    API_KEY = st.session_state.GOOGLE_API_KEY
    genai.configure(api_key=API_KEY)

    # Initialize models now that we have a key
    base_model = genai.GenerativeModel("gemini-1.5-flash-latest")
    langchain_chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7, google_api_key=API_KEY)
    embedding_model_name = "models/text-embedding-004"
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name, google_api_key=API_KEY)

except Exception as e:
    st.error(f"🔴 Error initializing Google AI. Please check your API Key. It might be invalid or have insufficient permissions. Details: {e}")
    st.stop()


# --- Main Chat Area ---
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
            final_image_data = st.session_state.get("current_chat_image_parts")
            response = get_gemini_response_text_image(
                TEXT_IMAGE_PROMPT, text_input=prompt, image_parts=final_image_data
            )
        elif st.session_state.app_mode == "📄 PDF Report Q&A":
            if st.session_state.pdf_processed:
                response = handle_pdf_query(prompt)
            else:
                response = "⚠️ Please upload and process PDF documents first using the sidebar."
        elif st.session_state.app_mode == "🎬 Video Analysis":
            video_ready = (st.session_state.get("video_processed") and
                           st.session_state.get("uploaded_video_uri") and
                           os.path.exists(st.session_state.uploaded_video_uri))
            if video_ready:
                message_placeholder.empty()
                response = get_gemini_response_video(
                    VIDEO_ANALYSIS_PROMPT, st.session_state.uploaded_video_uri, prompt
                )
            elif not st.session_state.get("uploaded_video_uri"):
                 response = "⚠️ Please upload a video using the sidebar first."
            else:
                response = "⚠️ Video not ready or file path invalid. Please re-upload if necessary."

        if response:
            message_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
             error_message = "Could not generate a response. Please check inputs/API key, or try again."
             message_placeholder.warning(error_message)
             st.session_state.messages.append({"role": "assistant", "content": error_message})

if st.session_state.app_mode == "📄 PDF Report Q&A":
    if not st.session_state.pdf_processed:
        st.info("ℹ️ Upload PDF reports via the sidebar & click 'Process'. Then ask questions.")
    else:
        st.success("✅ PDFs processed. Ask specific questions about the reports.")
elif st.session_state.app_mode == "🎬 Video Analysis":
     video_ready = (st.session_state.get("video_processed") and
                    st.session_state.get("uploaded_video_uri") and
                    os.path.exists(st.session_state.uploaded_video_uri))
     if not video_ready:
         st.info("ℹ️ Upload a video via the sidebar. Once ready, ask specific questions.")
     else:
         st.success(f"✅ Video '{os.path.basename(st.session_state.uploaded_video_uri)}' is ready. Ask questions about it.")
elif st.session_state.app_mode == "💬 General Chat & Image":
    st.info("ℹ️ Ask general fitness/health questions, or upload an image above for analysis.")
    
    
    











