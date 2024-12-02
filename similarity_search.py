import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Initialize Pinecone and connect to the index
api_key = "REDACTED"
course_description_index_name = "course-descriptions"

pc = Pinecone(api_key=api_key)
course_description_index = pc.Index(course_description_index_name)

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to query the Pinecone index and retrieve similar courses
def query_courses(user_interest, top_k=5):
    """
    Query the course-descriptions Pinecone index to find courses most similar to the user's interest.

    Parameters:
        user_interest (str): A sentence describing the user's interest.
        top_k (int): Number of top similar courses to retrieve.

    Returns:
        list: A list of courses ranked by similarity, including their course IDs and metadata.
    """
    try:
        # Generate embedding for user input
        user_embedding = model.encode(user_interest).tolist()

        # Query the Pinecone index
        results = course_description_index.query(
            vector=user_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # Format and return the results
        suggestions = []
        for match in results["matches"]:
            suggestions.append({
                "course_id": match["id"],  # Course ID
                "similarity_score": match["score"],  # Cosine similarity score
                "description": match["metadata"].get("description", "No description available"),
                "units": match["metadata"].get("units", "Unknown")
            })
        return suggestions

    except Exception as e:
        print(f"An error occurred during the query: {e}")
        return []

# Main function to handle user input and suggest courses
def suggest_courses():
    # Step 1: Get user input
    user_interest = input("Enter your sentence of interest: ").strip()
    if not user_interest:
        print("No interest sentence provided. Please try again.")
        return

    # Step 2: Query Pinecone for course suggestions
    print("\nSearching for courses...")
    suggestions = query_courses(user_interest, top_k=5)

    # Step 3: Display the results
    if not suggestions:
        print("No similar courses found based on your interest. Try a different input.")
        return

    print("\nRecommended Courses:")
    for suggestion in suggestions:
        print(f"Course ID: {suggestion['course_id']}")
        print(f"Similarity Score: {suggestion['similarity_score']:.4f}")
        print(f"Description: {suggestion['description']}")
        print(f"Units: {suggestion['units']}")
        print("---")

# Run the function
suggest_courses()
