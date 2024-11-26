import os
from pinecone import Pinecone, ServerlessSpec
import json 

# Set your Pinecone API key
api_key = "REDACTED" 

# Initialize Pinecone
pc = Pinecone(
    api_key=api_key,
)

# Create or connect to the index
index_name = "course-embeddings"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Match the embedding size (e.g., 384 for MiniLM)
        metric='cosine',  # Use cosine similarity or another metric
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # Replace with your actual region
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Load JSON Lines and upsert data into Pinecone
batch_size = 100
upserts = []
try:
    with open("vectorized_courses_input.jsonl", "r") as file:
        for line in file:
            try:
                record = json.loads(line.strip())
                if "course_name" in record and "embedding" in record and "category" in record:
                    upserts.append({
                        "id": record["course_name"],
                        "values": record["embedding"],
                        "metadata": {"category": record["category"]}
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

    # Final upsert
    if upserts:
        index.upsert(vectors=upserts)
        print(f"Upserted {len(upserts)} remaining records.")

    print(f"Data successfully upserted into index '{index_name}'.")

except Exception as e:
    print(f"An error occurred: {e}")