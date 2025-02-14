import os
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()


# Set API keys
PINECONE_API_KEY = "pcsk_2uKCF8_5LSz4hbio5WP681G6ThuJp3vBDxx7tuWSrM2RXrviFnwe7LmvEB5YVDGmm3mN5w"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
course_description_index_name = "course-descriptions"
course_description_index = pc.Index(course_description_index_name)

# Load Sentence Transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize OpenAI or use a free alternative (e.g., Hugging Face)
llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# Function to query Pinecone and retrieve similar courses
def query_courses(user_interest, top_k=5):
    """
    Query the Pinecone index for courses most relevant to the user's interest.
    """
    try:
        # Generate user input embedding
        user_embedding = model.encode(user_interest).tolist()

        # Query Pinecone
        results = course_description_index.query(
            vector=user_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # Format results
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
        print(f"An error occurred: {e}")
        return []

# Use LangChain to format the response using an LLM
def format_course_suggestions(suggestions, user_interest):
    """
    Use a language model to format and enhance the course recommendations.
    """
    prompt_template = PromptTemplate(
        input_variables=["user_interest", "course_list"],
        template="""
        Based on the user's interest in "{user_interest}", suggest relevant courses in a structured way.
        Here are some course matches:

        {course_list}

        Format the response in a professional yet engaging way, including course IDs and descriptions.
        """
    )

    course_list_str = "\n".join(
        [f"- **{s['course_id']}**: {s['description']} (Units: {s['units']})" for s in suggestions]
    )

    # Generate a response using the LLM
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run(user_interest=user_interest, course_list=course_list_str)

    return response

# Main function to interact with the user
def suggest_courses():
    """
    Get user input, query Pinecone, and use LLM to generate a response.
    """
    user_interest = input("Enter your interest: ").strip()
    if not user_interest:
        print("No interest provided. Please try again.")
        return

    print("\nSearching for relevant courses...")
    suggestions = query_courses(user_interest, top_k=5)

    if not suggestions:
        print("No matching courses found. Try a different input.")
        return

    # Generate formatted response
    formatted_response = format_course_suggestions(suggestions, user_interest)
    print("\n" + formatted_response)

# Run the function
suggest_courses()
