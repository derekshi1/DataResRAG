import json
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

with open("data_theory_req.json", "r") as file:
    data = json.load(file)
    
# Function to create embeddings for a course
def get_embedding(course_name):
    return model.encode(course_name)

# Flatten JSON and vectorize courses into a tabular structure
records = []
def vectorize_courses(data, category_path=""):
    for key, value in data.items():
        if isinstance(value, dict):  # Nested categories
            vectorize_courses(value, category_path=f"{category_path}/{key}" if category_path else key)
        elif isinstance(value, list):  # List of courses
            for course in value:
                embedding = get_embedding(course)
                records.append({
                    "category": category_path,
                    "course_name": course,
                    "embedding": embedding
                })

# Process and vectorize courses
vectorize_courses(data)

# Convert to a DataFrame for better handling and viewing
df = pd.DataFrame(records)

# Save DataFrame as a CSV or JSON for later use
df.to_csv("vectorized_courses.csv", index=False)
df.to_json("vectorized_courses.json", orient="records", lines=True)

print("Vectorization complete. Data saved to 'vectorized_courses.csv' and 'vectorized_courses.json'.")
