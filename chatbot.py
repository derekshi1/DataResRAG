# config.py
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_community.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()

# Set API keys
PINECONE_API_KEY = "pcsk_2uKCF8_5LSz4hbio5WP681G6ThuJp3vBDxx7tuWSrM2RXrviFnwe7LmvEB5YVDGmm3mN5w"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
course_description_index_name = "course-descriptions-combined"
course_description_index = pc.Index(course_description_index_name)

# Load Sentence Transformer model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize OpenAI LLM
llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)




# chatbot.py
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Streamlit Page Title
st.title("🤖 Chat with AI Advisor")
st.markdown("Ask about courses, prerequisites, industry topics, and more!")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")

# Function: Get Course Recommendations
def get_course_recommendations(query):
    """Queries Pinecone for relevant courses.
    - Mention similar courses or related topics
    - Avoid overly technical jargon
    - Be interactive with the user (ask if they have any further questions about the topic, ask followup questions for clarity if needed)
    - Keep responses at least 1-2 sentences"""
    user_embedding = model.encode(query).tolist()
    results = course_description_index.query(vector=user_embedding, top_k=3, include_metadata=True)

    if not results["matches"]:
        return "⚠️ No matching courses found."

    recommendations = []
    for match in results["matches"]:
        course_id = match["id"]
        description = match["metadata"].get("description", "No description available")
        recommendations.append(f"📘 **{course_id}:** {description}")
    
    return "\n\n".join(recommendations)

course_tool = Tool(
    name="Course Search",
    func=get_course_recommendations,
    description="Retrieve the best course recommendations based on user interests."
)

# Function: Fetch Prerequisites for a Course
def get_course_prerequisites(course_name):
    """Fetch prerequisites for a given course."""
    results = course_description_index.query(
        vector=model.encode(course_name).tolist(),
        top_k=1,
        include_metadata=True
    )
    if results["matches"]:
        requisites = results["matches"][0]["metadata"].get("requisites", "No prerequisites listed.")
        return f"📘 **{course_name} Prerequisites:** {requisites}"
    return f"⚠️ No prerequisites found for {course_name}."

prereq_tool = Tool(
    name="Prerequisite Checker",
    func=get_course_prerequisites,
    description="Fetch prerequisites for a given course."
)

# Function: Explain Industry Topics
def explain_topic(topic):
    """Uses LLM to explain industry topics concisely."""
    explanation_prompt = f"""
    Explain the topic "{topic}" concisely:
    - Key insights, and justified responses
    - Real-world applications
    - Mention similar courses or related topics
    - Avoid overly technical jargon
    - Be interactive with the user (ask if they have any further questions about the topic, ask followup questions for clarity if needed)
    - Keep responses at least 1-2 sentences
    """
    chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["topic"], template=explanation_prompt))
    return chain.run(topic=topic)

industry_tool = Tool(
    name="Industry Explainer",
    func=explain_topic,
    description="Explain industry topics (e.g., AI, Data Science, Machine Learning)."
)

# Initialize AI Chatbot Agent
if "chatbot" not in st.session_state:
    st.session_state.chatbot = initialize_agent(
        tools=[course_tool, prereq_tool, industry_tool],
        llm=llm,
        memory=st.session_state.memory,
        agent="zero-shot-react-description",
        verbose=False,
    )

# Chat Interface in a Container
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask about courses, prerequisites, or industry topics...")

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process chatbot response
    with st.spinner("🤖 Thinking..."):
        response = st.session_state.chatbot.run(user_input)

    # Display assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
