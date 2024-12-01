import os
from pinecone import Pinecone, ServerlessSpec
import json

api_key = "REDACTED" 

# Initialize Pinecone
pc = Pinecone(api_key=api_key)

# Create or connect to the index for descriptions
index_name = "course-descriptions"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Match the embedding size of the model used (e.g., MiniLM)
        metric='cosine',  # Use cosine similarity or another metric
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Load JSON Lines file containing vectorized course descriptions
batch_size = 100
upserts = []
try:
    with open("vectorized_courses_descriptions.jsonl", "r") as file:
        for line in file:
            try:
                record = json.loads(line.strip())
                # Check if all required fields are present
                if "course_id" in record and "embedding" in record and "description" in record:
                    upserts.append({
                        "id": record["course_id"],  # Unique course ID
                        "values": record["embedding"],  # Embedding vector
                        "metadata": {
                            "description": record["description"],  # Course description
                            "units": record.get("units", "Unknown")  # Optional units
                        }
                    })
                else:
                    print(f"Missing keys in record: {record}")

                # Perform upserts in batches
                if len(upserts) == batch_size:
                    index.upsert(vectors=upserts)
                    print(f"Upserted {len(upserts)} records.")
                    upserts = []  # Clear the batch

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {e}")

    # Final upsert for remaining records
    if upserts:
        index.upsert(vectors=upserts)
        print(f"Upserted {len(upserts)} remaining records.")

    print(f"Data successfully upserted into index '{index_name}'.")

except Exception as e:
    print(f"An error occurred: {e}")
