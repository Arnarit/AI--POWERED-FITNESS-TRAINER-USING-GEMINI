from dotenv import load_dotenv
load_dotenv()  # loading all the environments variables

import streamlit as st
import os
from PIL import Image
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# FUNCTION TO LOAD GEMINI MODEL AND GET RESPONSE (Keep this as is, it's core logic)
model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")


# --- CSS Styling (Enhanced for Kimi-like aesthetics) ---
st.markdown("""
<style>
body {
    color: #FFFFFF;
    background-color: #0e1117;  /* Dark, sophisticated background */
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; /* Modern, clean font */
}
.stApp {
    background: linear-gradient(135deg, #1a1a1a, #0e1117); /* Subtle gradient */
}

/* Chat Interface */
.chat-container {
    display: flex;
    flex-direction: column;
    max-height: 70vh;  /* Limit height, add scroll */
    overflow-y: auto;
    padding: 10px;
    border: 1px solid #4CAF50;  /* Add border */
    border-radius: 8px;          /* Rounded corners */
}

.user-message {
    background-color: rgba(76, 175, 80, 0.2); /* Light green background */
    color: #FFFFFF;
    padding: 8px 12px;
    border-radius: 15px 15px 15px 3px;     /* Rounded corners, different for user */
    margin-bottom: 8px;
    align-self: flex-end; /* Align to the right */
    max-width: 70%;       /* Limit width */
    word-wrap: break-word;  /* Wrap long words */
}

.ai-message {
    background-color: rgba(40, 40, 40, 0.9);
    color: #FFFFFF;
    padding: 8px 12px;
    border-radius: 15px 15px 3px 15px;    /* Rounded corners, different for AI */
    margin-bottom: 8px;
    align-self: flex-start;  /* Align to the left */
    max-width: 70%;
    word-wrap: break-word;
}


/* Headers */
h1, h2, h3 {
    color: #4CAF50;  /* Vibrant green, but toned down */
    font-weight: 600; /* Semi-bold */
    margin-bottom: 0.5em;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); /* Adds depth */
}

/* Sidebar */
.sidebar .sidebar-content {
    background-color: rgba(26, 26, 26, 0.8); /* Slightly transparent */
    backdrop-filter: blur(10px); /* Glassmorphism effect */
    border-right: 1px solid #4CAF50;
}

/* Buttons */
.stButton>button {
    background-color: #4CAF50;
    color: #FFFFFF;
    border-radius: 25px; /* Even more rounded */
    border: none;
    padding: 12px 24px; /* Larger padding */
    font-size: 1.1em;
    font-weight: 500;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.4); /* Shadow for depth */
    transition: all 0.3s ease;
    width: 100%;  /* Make buttons full width */
    margin-bottom: 5px;
}
.stButton>button:hover {
    background-color: #66BB6A; /* Lighter green on hover */
    transform: translateY(-2px); /* Slight lift on hover */
    box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.5);
}
.stButton>button:active {
    transform: translateY(0px); /* Reset position on click */
    box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.3);
}

/* Text Area (for input)*/
.stTextInput>div>div>input {
    background-color: rgba(40, 40, 40, 0.9);
    color: #FFFFFF;
    border: 1px solid #4CAF50;
    border-radius: 8px;
    padding: 10px;
    box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.3);
    transition: border-color 0.3s ease;
    width: calc(100% - 48px); /*Adjust for send button*/
}

.stTextInput>div>div>input:focus{
     border-color: #66BB6A; /* Highlight on focus */
}


/* File Uploader */
.stFileUploader {
    background-color: rgba(30, 30, 30, 0.8);
    border: 3px dashed #4CAF50;
    border-radius: 15px; /* More rounded */
    padding: 25px;
}

/* Markdown */
.stMarkdown {
    color: #DDDDDD; /* Light grey for better readability */
    line-height: 1.6;
}

/* Progress & Spinner */
.stProgress .st-bo {
    background-color: #4CAF50;
}
.stSpinner {
    color: #4CAF50;
}

/* Success Message */
.st-success {
    background-color: rgba(25, 25, 25, 0.9);
    color: #4CAF50;
    border: 1px solid #4CAF50;
    padding: 12px;
    border-radius: 8px;
    font-weight: 500;
}

/* Error & Warning Messages */
.st-error, .st-warning {
     background-color: rgba(100, 20, 20, 0.8); /* Dark red/orange for errors */
     color: #FFFFFF;
     border: 1px solid #FF5722;
     padding: 12px;
     border-radius: 8px;
     font-weight: 500;
}
.st-warning{
     background-color: rgba(100, 70, 20, 0.8); /* Dark red/orange for errors */
}

/* Image */
.stImage>div {
    border-radius: 10px;  /* Rounded image corners */
    overflow: hidden;   /* Ensures rounded corners work */
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.4); /* Image shadow */
}
.stImage img {
    display: block; /* Remove extra space under image */
    margin: 0 auto; /* Center the image horizontally */
}

/* Columns - Add subtle borders */
[data-testid="column"] {
    border: 1px solid rgba(76, 175, 80, 0.3); /* Subtle green border */
    border-radius: 8px;
    padding: 10px;
    margin: 5px;
    background-color: rgba(26, 26, 26, 0.1);  /*Very slight background*/
}

/* Input container for chat */
.input-container {
    display: flex;
    align-items: center; /* Align items vertically */
    gap: 10px;          /* Spacing between input and button */
    padding: 10px;
}


</style>
""", unsafe_allow_html=True)


# --- Session State Management (for chat history and page) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_page" not in st.session_state:
    st.session_state.current_page = 1
if "processed" not in st.session_state:
    st.session_state.processed = False

# --- Navigation Functions ---
def next_page():
    st.session_state.current_page = 2
    st.session_state.messages = []  # Clear chat on page change
    st.session_state.processed = False

def previous_page():
    st.session_state.current_page = 1
    st.session_state.messages = []

# --- Langchain Helper Functions (Keep, but adapt for chat) ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    As an fitness trainers you are expert in understanding healthcare domain including medical, medicines, fitness, sports and Brahmacharya.
    You have extreme deep knowledge of medical sciences, Brahmacharya, physiological and psychological fitness, spiritual and ayurveda.
    We will ask you many questions on health domain and you will have to answer any questions.
    Answer the question as detailed as possible from the provided context, make sure to provide all the details:
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


    If question is not related to health domain then say just this line : "Sorry, I am an AI fitness trainer, I can only answer questions
    related to health domain. Please ask a question related to health domain."

    if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer should be in between 1000 to 10000 numbers of words:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response["output_text"]  # Return the string, not the whole dict

def get_gemini_response2(image, input_prompt2, text_input):
    response = model.generate_content([image[0], input_prompt2, text_input])
    return response.text

def get_gemini_response1(input_prompt1, text_input):
    response = model.generate_content([input_prompt1, text_input])
    return response.text

def input_image_details(uploaded_file):
    if uploaded_file is not None:  # Check for None here
      bytes_data = uploaded_file.getvalue()
      image_parts = [
          {
              "mime_type": uploaded_file.type,
              "data": bytes_data
          }
      ]
      return image_parts
    return None  # Return None if no file


# --- UI Components (Adapted for chat) ---

# --- Page 1: Image/Text Input ---
if st.session_state.current_page == 1:
    st.title("AI Fitness Trainer 🧘‍♂️")
    st.markdown("#### Your Personalized Wellness Companion")
    st.markdown("---")

    # Display chat history
    chat_container = st.container()
    with chat_container:
      for message in st.session_state.messages:
          with st.chat_message(message["role"]):
              st.markdown(message["content"])

    # Input area at the bottom - WRAPPED IN FORM
    with st.form(key='my_form_1'):
        col1, col2 = st.columns([7,1])
        with col1:
            text_input = st.text_area("Describe your health concerns, goals, or any questions you have:", key="text_input1", label_visibility="collapsed" , height=200)
        with col2:
           submit_text = st.form_submit_button("Send")
        with col1:
          uploaded_file = st.file_uploader("Upload a relevant image (optional)", type=["jpeg", "jpg", "png"],key="uploader1")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)


    with st.sidebar:
        st.markdown("### 🔗 Quick Actions")
        submit1 = st.button("Upload Medical/Fitness Reports (PDF)")
        if submit1:
            next_page()

        st.divider()
        st.markdown("### ✨ Model Capabilities")
        st.markdown("""
        - 🏋️ Personalized Fitness Recommendations
        - 🍎 Nutritional Guidance
        - 🧠 Mental Wellness Support
        - 🔬 Deep Medical Report Analysis
        - 🗣️ Natural Language Understanding
        - 👀 Advanced Image Recognition
        - 📚 Contextual Answers (RAG)
        """)

    input_prompt1 = """
    As an fitness trainers you are expert in understanding healthcare domain including medical, medicines, fitness, sports and Brahmacharya.
    You have extreme deep knowledge of medical sciences, Brahmacharya, physiological and psychological fitness, spiritual and ayurveda.
    We will ask you many questions on health domain and you will have to answer any questions
    of user in medical, scientific and evidence-based manner in between 1000 to 10000 numbers of words.

    Disclaimer: This AI Fitness Trainer application provides information and recommendations for general fitness and wellness purposes only.
    It is not intended legally,and should not be used as, a substitute for professional medical advice, diagnosis, or treatment.
    Always seek the advice of your physician or other qualified health provider physically with any questions you may have regarding
    a medical condition for getting better benefits.
    Never disregard professional medical advice or delay in seeking it because of something you have read or received through this application.


    If question is not related to health domain then say just this line : "Sorry, I am an AI fitness trainer, I can only answer questions
    related to health domain. Please ask a question related to health domain."

    if the answer is not in provided user input just say, "answer is not available in the user input", don't provide the wrong answer
    """


    input_prompt2 = """
    As an fitness trainers you are expert in understanding healthcare domain including medical, medicines, fitness, sports and Brahmacharya.
    You have extreme deep knowledge of medical sciences, Brahmacharya, physiological and psychological fitness, spiritual and ayurveda.
    We will ask you many questions on health domain and you will have to answer any questions based on the any types of
    resolutions of uploaded image in medical, scientific and evidence-based in between 1000 to 10000 numbers of words.

    Disclaimer: This AI Fitness Trainer application provides information and recommendations for general fitness and wellness purposes only.
    It is not intended legally,and should not be used as, a substitute for professional medical advice, diagnosis, or treatment.
    Always seek the advice of your physician or other qualified health provider physically with any questions you may have regarding
    a medical condition for getting better benefits.
    Never disregard professional medical advice or delay in seeking it because of something you have read or received through this application.


    If question is not related to health domain then say just this line : "Sorry, I am an AI fitness trainer, I can only answer questions
    related to health domain. Please ask a question related to health domain."

    if the answer is not in provided uploaded images just say, "answer is not available in the uploaded images", don't provide the wrong answer
    """

    if submit_text or uploaded_file:  # Respond to either button
      if text_input:
            st.session_state.messages.append({"role": "user", "content": text_input})
            with st.chat_message("user"):
                st.markdown(text_input)
      if uploaded_file:
          image_data = input_image_details(uploaded_file)
          if image_data is not None:
              with st.spinner("Processing image..."):
                   response_image = get_gemini_response2(image_data, input_prompt2, text_input if text_input else "")
              st.session_state.messages.append({"role": "assistant", "content": response_image})
              with st.chat_message("assistant"):
                st.markdown(response_image)

      elif text_input:
        with st.spinner("Thinking..."):

            response = get_gemini_response1(input_prompt1, text_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)




# --- Page 2: PDF Document Q&A ---
elif st.session_state.current_page == 2:
    st.title("AI Fitness Trainer 🧘‍♂️")
    st.markdown("#### Ask Questions About Your Reports")
    st.markdown("---")


    # Chat History Display (Same as page 1)
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Input and Buttons - WRAPPED IN FORM
    with st.form(key='my_form_2'):
        col1, col2 = st.columns([7, 1])
        with col1:
          user_question = st.text_area("Describe your health concerns, goals, or any questions you have:", key="text_input1", label_visibility="collapsed" , height=200)
        with col2:
          submit2 = st.form_submit_button("Ask")
        with col2:
           st.form_submit_button("Back", on_click=previous_page)  # Back button in form too


    with st.sidebar:
        st.markdown("### ⬆️ Upload & Process")
        pdf_docs = st.file_uploader(
            "Upload your Medical and Fitness reports (PDF format)",
            accept_multiple_files=True
        )
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Analyzing Your Documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.session_state.processed = True
                    st.success("✅ Documents processed! Ask away!")
            else:
                st.error("Please upload at least one PDF.")


    if submit2:
        if st.session_state.processed:
            if user_question:
                st.session_state.messages.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)

                with st.spinner("Thinking..."):
                    response = user_input(user_question)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                      st.markdown(response)
            else:
                st.warning("Please enter a question.")
        else:
            st.warning("Please process documents first.")


