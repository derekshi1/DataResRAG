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
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.tools import Tool
from dotenv import load_dotenv
import time
import re


load_dotenv()


# Set API keys
PINECONE_API_KEY = "pcsk_2uKCF8_5LSz4hbio5WP681G6ThuJp3vBDxx7tuWSrM2RXrviFnwe7LmvEB5YVDGmm3mN5w"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
course_description_index_name = "course-descriptions-combined"
course_description_index = pc.Index(course_description_index_name)


# Load Sentence Transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')


llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)


st.title("📚 AI-Powered Course Recommendation")
st.subheader("Find the best courses based on your interests!")


user_interest = st.text_input("Enter your academic interest (e.g., AI, Data Science, Psychology):")


def get_progress_bar_html(percentage):
   """
   Generates an HTML-based progress bar with a dynamic color gradient.
   - Green (High Match) → Yellow (Medium Match) → Red (Low Match)
   """
   color = f"rgb({255 - int(2.55 * percentage)}, {int(2.55 * percentage)}, 50)"  # Dynamic RGB color


   return f"""
   <div style="width: 100%; background-color: #eee; border-radius: 5px; height: 15px; position: relative; margin-bottom: 5px;">
       <div style="width: {percentage}%; height: 100%; background-color: {color}; border-radius: 5px;"></div>
   </div>
   """


def query_courses(user_interest, top_k=5):
   """
   Query the Pinecone index for courses most relevant to the user's interest and use LLM to extract formatted details.
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
           percentage_match = round(match["score"] * 100, 2)
           description = match["metadata"].get("description", "No description available")


           # Send to LLM for structured formatting
           formatting_prompt = f"""
           Extract structured details from the following course description.


           Course Description:
           {description}


           Please format the output with:
           - **Requisites:** List all prerequisite courses (if none, say "None").
           - **Units:** Extract the number of units (if unknown, say "Unknown").
           - **Summary:** Summarize the course into bullet points focusing on key topics.
           """


           formatting_response = llm.invoke(formatting_prompt)
           formatted_details = formatting_response.content.strip().split("\n")


           # Parse LLM response
           requisite_courses = "None"
           extracted_units = "Unknown"
           summary_lines = []


           for line in formatted_details:
               if line.startswith("**Requisites:**"):
                   requisite_courses = line.replace("**Requisites:**", "").strip()
               elif line.startswith("**Units:**"):
                   extracted_units = line.replace("**Units:**", "").strip()
               else:
                   summary_lines.append(line)


           summary_text = "\n".join(summary_lines)


           reasoning_prompt = f"""
           The user is interested in "{user_interest}". The following course has been matched:
           - **Course ID:** {match["id"]}
           - **Description Summary:** {summary_text}
           - **Match Score:** {percentage_match}%
           - **Units:** {extracted_units}
          
           Explain in a concise way why this course is a good match for the user’s interest. If the match is not good, then say that it's not a good match. Be specific about subtopics in the course description and how they relate to the user's input. Make the reasoning no more than 4 bullet points. Do not give trivial information. Make sure to be specific on how aspects of the course description relate to the user's input. Vary sentence structure and do not explicitly mention user input - focus on the actual input.
           """


           reasoning_response = llm.invoke(reasoning_prompt)
           reasoning_text = str(reasoning_response.content).strip()


           suggestions.append({
               "course_id": match["id"],
               "similarity_score": match["score"],
               "percentage_match": percentage_match,
               "description": summary_text, 
               "requisites": requisite_courses, 
               "units": extracted_units, 
               "metadata": {"reasoning": reasoning_text}  # LLM-generated reasoning
           })


       return suggestions


   except Exception as e:
       st.error(f"An error occurred: {e}")
       return []




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
       st.markdown("## 🎓 Recommended Courses:")


       for s in suggestions:
           st.markdown(f"### {s['course_id']} ({s['percentage_match']}% match)")
           st.markdown(get_progress_bar_html(s["percentage_match"]), unsafe_allow_html=True)   
           st.markdown("📖 **Summary:**")
           st.markdown(s["description"])


           # Course Requisites
           if s['requisites'] != "None":
               st.markdown(f"📝 **Requisites:** {s['requisites']}") 




           st.markdown("🧠 **Why this course is recommended:**")
           st.markdown(s['metadata']['reasoning'])
           st.markdown("---")
       else:
           st.warning("⚠️ No matching courses found. Try a different input.")





