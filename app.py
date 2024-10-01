import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

st.set_page_config(page_title="AIChatbot", layout="centered")

st.markdown(
    """
    <h1 style="color: #008000; text-align: center;">
        <div style="display: flex; align-items: center; justify-content: center;">   
            AI Chatbot
            <img src="https://www.pngfind.com/pngs/m/126-1269385_chatbots-builder-pricing-crozdesk-chat-bot-png-transparent.png" alt="Chatbot Logo" width="80" height="80" style="margin-right: 10px;">
        </div>
    </h1>
    """, unsafe_allow_html=True
)

@st.cache_resource
def load_or_create_faiss():
    if os.path.exists("faiss_index"):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        return db
    else:
        loader = PyPDFLoader('Medical_book.pdf')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
        docs = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        db = FAISS.from_documents(docs, embeddings)
        db.save_local("faiss_index") 
        return db

db = load_or_create_faiss()

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.5, token=hf_token
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

history = StreamlitChatMessageHistory(key="chat_messages")

def handle_special_questions(question):
    if "invented you" in question.lower() or "created you" in question.lower() or "developed you" in question.lower():
        return "I was created by Bharathkumar M S."
    return None

st.markdown(
    "<h4 style='color:red;'>I am an AI chatbot. Ask anything you want:</h4>", 
    unsafe_allow_html=True
)

user_question = st.chat_input("Your Question")

if user_question:
    st.chat_message("human").write(user_question)
    
    special_response = handle_special_questions(user_question)
    if special_response:
        st.chat_message("ai").write(special_response)  
    else:
        with st.spinner('Processing your question...'):
            context_docs = db.similarity_search(user_question, k=2)  
            context = "\n".join([doc.page_content for doc in context_docs])

            history.add_user_message(user_question)

            past_messages = history.messages
            
            response = llm_chain.run(user_question)

            history.add_ai_message(response)  

            st.chat_message("ai").write(response)

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
        padding: 10px 0;
        z-index: 1000;  /* Ensure the footer is on top of other content */
        display: flex; /* Use flexbox for alignment */
        justify-content: space-between; /* Space between elements */
        align-items: center; /* Center vertically */
        padding: 10px 20px; /* Add padding */
    }
    .footer img {
        vertical-align: middle;
        margin-left: 10px;
    }
    </style>
    <div class="footer">
        <div>
            Developed by Bharathkumar M S
            <a href="https://www.linkedin.com/in/bharathkumar-m-s/" target="_blank">
                <img src="https://upload.wikimedia.org/wikipedia/commons/e/e9/Linkedin_icon.svg" alt="LinkedIn" width="30" height="30">
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True
)
