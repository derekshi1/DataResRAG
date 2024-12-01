import json
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the JSON file containing combined courses data
with open("combined_course_data.json", "r") as file:
    data = json.load(file)

# Function to create embeddings for a course description
def get_embedding(text):
    return model.encode(text)

# Flatten JSON and vectorize courses and descriptions into a tabular structure
records = []

def vectorize_courses(data):
    for course in data:
        # Extract information from each course entry
        course_id = course.get("course_id", "Unknown ID")
        description = course.get("description", "No description available")
        category = course.get("category", "Unknown category")
        sequence = course.get("sequence", "None")
        
        # Generate embedding for the description
        embedding = get_embedding(description)
        
        # Add the record to the list
        records.append({
            "course_id": course_id,
            "description": description,
            "category": category,
            "sequence": sequence,
            "embedding": embedding
        })

# Process and vectorize courses and descriptions
vectorize_courses(data)

# Convert to a DataFrame for better handling and viewing
df = pd.DataFrame(records)

# Save DataFrame as a CSV or JSON for later use
df.to_csv("vectorized_combined_courses.csv", index=False)
df.to_json("vectorized_combined_courses.jsonl", orient="records", lines=True)

print("Vectorization complete. Data saved to 'vectorized_combined_courses.csv' and 'vectorized_combined_courses.jsonl'.")
