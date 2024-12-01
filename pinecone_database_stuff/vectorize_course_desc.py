import json
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the JSON file containing courses and descriptions
with open("pinecone_database_stuff/dt_course_descriptions.json", "r") as file:
    data = json.load(file)

# Function to create embeddings for a course description
def get_embedding(text):
    return model.encode(text)

# Flatten JSON and vectorize courses and descriptions into a tabular structure
records = []

def vectorize_courses(data):
    for course_id, description_text in data.items():
        # Split the description into parts
        if "Description:" in description_text:
            description = description_text.split("Description:")[1].split("Units:")[0].strip()
            units = description_text.split("Units:")[-1].strip()
        else:
            description = "No description available"
            units = "Unknown"
        
        # Generate embedding for the description
        embedding = get_embedding(description)
        
        # Add the record to the list
        records.append({
            "course_id": course_id,
            "description": description,
            "units": units,
            "embedding": embedding
        })

# Process and vectorize courses and descriptions
vectorize_courses(data)

# Convert to a DataFrame for better handling and viewing
df = pd.DataFrame(records)

# Save DataFrame as a CSV or JSON for later use
df.to_csv("vectorized_courses_descriptions.csv", index=False)
df.to_json("vectorized_courses_descriptions.jsonl", orient="records", lines=True)

print("Vectorization complete. Data saved to 'vectorized_courses_descriptions.csv' and 'vectorized_courses_descriptions.jsonl'.")
