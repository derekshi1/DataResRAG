import os
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time

load_dotenv()


# Set API keys
PINECONE_API_KEY = "REDACTED"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
course_description_index_name = "course-descriptions"
course_description_index = pc.Index(course_description_index_name)

# Load Sentence Transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize OpenAI or use a free alternative (e.g., Hugging Face)
llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

st.title("📚 AI-Powered Course Recommendation")
st.subheader("Find the best courses based on your interests!")

# User input
user_interest = st.text_input("Enter your academic interest (e.g., AI, Data Science, Psychology):")

# Function to query Pinecone
def query_courses(user_interest, top_k=5):
    """
    Query the Pinecone index for courses most relevant to the user's interest.
    """
    try:
        user_embedding = model.encode(user_interest).tolist()

        results = course_description_index.query(
            vector=user_embedding,
            top_k=top_k,
            include_metadata=True
        )

        suggestions = []
        for match in results["matches"]:
            suggestions.append({
                "course_id": match["id"],
                "similarity_score": match["score"],
                "description": match["metadata"].get("description", "No description available"),
                "units": match["metadata"].get("units", "Unknown")
            })
        return suggestions

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []

# Format course suggestions using LangChain
def format_course_suggestions(suggestions, user_interest):
    """
    Use a language model to format and enhance course recommendations.
    """
    prompt_template = PromptTemplate(
        input_variables=["user_interest", "course_list"],
        template="""
        Based on the user's interest in "{user_interest}", suggest relevant courses in a structured way.
        Here are some course matches:

        {course_list}

        Format the response professionally while making it engaging.
        """
    )

    course_list_str = "\n".join(
        [f"- **{s['course_id']}**: {s['description']} (Units: {s['units']})" for s in suggestions]
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run(user_interest=user_interest, course_list=course_list_str)

    return response

# Button to get recommendations
if st.button("🔍 Find Courses"):
    if user_interest:
        with st.spinner("🔎 Searching for the best courses..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Fake loading animation for effect
            for percent in range(0, 101, 10):
                time.sleep(0.3)  # Simulate search time
                progress_bar.progress(percent)
                if percent < 30:
                    status_text.write("📡 Connecting to AI models...")
                elif percent < 60:
                    status_text.write("📊 Analyzing course descriptions...")
                elif percent < 90:
                    status_text.write("🧠 Applying smart recommendations...")
                else:
                    status_text.write("🚀 Almost done!")

            progress_bar.empty()  # Remove progress bar when done
            status_text.empty()

        # Fetch course suggestions
        suggestions = query_courses(user_interest, top_k=5)

        if suggestions:
            formatted_response = format_course_suggestions(suggestions, user_interest)
            st.markdown("## 🎓 Recommended Courses:")
            st.markdown(formatted_response)
        else:
            st.warning("⚠️ No matching courses found. Try a different input.")
    else:
        st.error("🚨 Please enter a valid interest.")