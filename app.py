import streamlit as st
import re
import os
from groq import Groq
from crewai import Agent, Task, Crew, Process, LLM
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage

# Set up the page configuration
st.set_page_config(
    page_title="CDP How-To Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session states
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'cdp_docs' not in st.session_state:
    st.session_state.cdp_docs = {}

# Groq API key input
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
os.environ["GROQ_API_KEY"] = ""

# Create Groq client when API key is provided
groq_client = None
if api_key:
    groq_client = Groq(api_key=api_key)

# CDP Documentation URLs
CDP_DOCS = {
    "Segment": "https://segment.com/docs/?ref=nav",
    "mParticle": "https://docs.mparticle.com/",
    "Lytics": "https://docs.lytics.com/",
    "Zeotap": "https://docs.zeotap.com/home/en-us/"
}

# Cache function for document loading and indexing
@st.cache_resource
def load_and_index_docs(url, api_key):
    if not api_key:
        return None
    
    try:
        # Load documents from the URL
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store with HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        return vectorstore
    except Exception as e:
        st.error(f"Error loading documentation: {str(e)}")
        return None

def query_groq(messages, model, temperature=0):
    if not groq_client:
        return "Error: Groq client not initialized. Please check your API key."
    
    try:
        # Convert LangChain message format to Groq format if needed
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})
            else:
                formatted_messages.append(msg)

        # Ensure the model name is passed as a string
        response = groq_client.chat.completions.create(
        model=model,
        messages=formatted_messages,
        temperature=temperature,
        max_tokens=3000,
        stream=False  # Disable streaming to avoid excessive token usage
    )

        return response.choices[0].message.content
    except Exception as e:
        return f"Error querying Groq: {str(e)}"

# Class to replace LangChain LLM with direct Groq integration
class GroqLLM:
    def __init__(self, model, temperature=0):
        self.model = model
        self.temperature = temperature
    
    def invoke(self, prompt):
        # Directly pass the model name and prompt in the query
        messages = [{"role": "user", "content": prompt}]
        return query_groq(messages, self.temperature, model=self.model)

# Main header
st.title("CDP How-To Assistant")
st.markdown("Ask me how to perform tasks in Segment, mParticle, Lytics, or Zeotap")


# Initialize DocumentRetrievalAgent
def create_retrieval_agent(cdp_name, vectorstore):
    if not vectorstore:
        return None
    
    search = DuckDuckGoSearchRun()
    
    # Create a retrieval tool
    retrieval_tool = Tool(
        name=f"{cdp_name}DocsRetrieval",
        description=f"Search the {cdp_name} documentation for information.",
        func=lambda query: vectorstore.similarity_search(query, k=3)
    )
    
    # Create a web search tool
    search_tool = Tool(
        name="WebSearch",
        description="Search the web for additional information if not found in docs.",
        func=search.run
    )
    
    retrieval_agent = Agent(
        role=f"{cdp_name} Documentation Expert",
        goal=f"Find the most relevant information in {cdp_name} documentation.",
        backstory=f"You are an expert in {cdp_name} CDP and have deep knowledge of its documentation.",
        verbose=True,
        allow_delegation=False,
        tools=[retrieval_tool, search_tool],
        llm=LLM(api_key=api_key, model="groq/gemma-7b-it", temperature=0, max_tokens=100)
    )
    
    return retrieval_agent

# Initialize AnswerGenerationAgent
def create_answer_agent():
    answer_agent = Agent(
        role="CDP How-To Expert",
        goal="Provide clear, accurate answers to CDP how-to questions",
        backstory="You are an expert in all major Customer Data Platforms. You provide step-by-step guidance on how to perform tasks in these platforms.",
        verbose=True,
        allow_delegation=True,
        llm=LLM(api_key=api_key, model="groq/gemma-7b-it", temperature=0, max_tokens=100)
    )
    
    return answer_agent

# Document loading status in sidebar
with st.sidebar:
    st.subheader("Documentation Status")
    
    model_selection = st.selectbox(
        "Select Groq Model",
        [
            "gemma-7b-it", 
            "mixtral-8x7b-32768",
            "gemma-7b-it"
        ],
        index=0
    )
    api_key = ""
    if api_key:
        for cdp, url in CDP_DOCS.items():
            if cdp not in st.session_state.cdp_docs or st.session_state.cdp_docs[cdp] is None:
                with st.spinner(f"Loading {cdp} documentation..."):
                    vectorstore = load_and_index_docs(url, api_key)
                    st.session_state.cdp_docs[cdp] = vectorstore
            
            if cdp in st.session_state.cdp_docs and st.session_state.cdp_docs[cdp] is not None:
                st.success(f"{cdp}: Loaded ✓")
            else:
                st.error(f"{cdp}: Not loaded ✗")
    else:
        st.warning("Enter your Groq API key to load documentation")

# Chat interface
question = st.chat_input("Ask a how-to question about Segment, mParticle, Lytics, or Zeotap")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question and api_key:
    # Add user question to chat history
    st.session_state.chat_history.append({"role": "user", "content": question})
    
    # Display user question
    with st.chat_message("user"):
        st.write(question)
    
    # Display assistant response with a loading spinner
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("Thinking..."):
            # Initialize agents and crew
            answer_agent = create_answer_agent()
            
            # Identify which CDP(s) the question is about
            cdp_pattern = re.compile(r'segment|mparticle|lytics|zeotap', re.IGNORECASE)
            mentioned_cdps = cdp_pattern.findall(question.lower())
            
            # If no CDP is explicitly mentioned, assume the question is about all CDPs or determine from context
            if not mentioned_cdps:
                # Check if the question is relevant to CDPs at all
                if not any(keyword in question.lower() for keyword in ['cdp', 'customer data', 'platform', 'data', 'integration', 'source', 'user', 'audience', 'segment']):
                    response = "I'm specialized in answering questions about Customer Data Platforms (CDPs) like Segment, mParticle, Lytics, and Zeotap. Could you please ask a question related to these platforms?"
                    response_placeholder.write(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.stop()
                else:
                    # If it's a general CDP question, include all CDPs
                    mentioned_cdps = ['segment', 'mparticle', 'lytics', 'zeotap']
            
            # Create a unique list of CDPs (converting to proper case)
            cdps_to_query = list(set([cdp.capitalize() if cdp != 'mparticle' else 'mParticle' for cdp in mentioned_cdps]))
            
            # Create retrieval agents for each mentioned CDP
            retrieval_agents = []
            for cdp in cdps_to_query:
                if cdp in st.session_state.cdp_docs and st.session_state.cdp_docs[cdp] is not None:
                    agent = create_retrieval_agent(cdp, st.session_state.cdp_docs[cdp])
                    if agent:
                        retrieval_agents.append(agent)
            
            if not retrieval_agents:
                response = "I couldn't load the necessary documentation to answer your question. Please make sure your API key is correct and try again."
                response_placeholder.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.stop()
            
            # Create tasks for each retrieval agent
            retrieval_tasks = []
            for i, agent in enumerate(retrieval_agents):
                cdp = cdps_to_query[i]
                task = Task(
                    description=f"Search {cdp} documentation for information about: {question}",
                    agent=agent,
                    expected_output=f"Detailed information from {cdp} documentation about how to accomplish the task."
                )
                retrieval_tasks.append(task)
            
            # Create the answer generation task
            answer_task = Task(
                description=f"Based on the retrieved information, answer the user's question: {question}",
                agent=answer_agent,
                expected_output="A clear, step-by-step answer to the user's question about how to accomplish a task in the CDP(s)."
            )
            
            # Create and run the crew
            crew = Crew(
                agents=[answer_agent] + retrieval_agents,
                tasks=retrieval_tasks + [answer_task],
                verbose=True,
                process=Process.sequential
            )
            
            try:
                result = crew.kickoff()
                response_placeholder.write(result)
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": result})
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                response_placeholder.write(error_message)
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})

elif question and not api_key:
    # Display a message if API key is missing
    with st.chat_message("assistant"):
        st.write("Please enter your Groq API key in the sidebar to use the chatbot.")

# Add instructions about installation requirements
with st.sidebar:
    st.markdown("---")
    st.subheader("Installation Requirements")
    st.code("pip install streamlit crewai langchain groq faiss-cpu sentence-transformers")
