import os
from pinecone import Pinecone, ServerlessSpec

# Set your Pinecone API key
api_key = "pcsk_2uKCF8_5LSz4hbio5WP681G6ThuJp3vBDxx7tuWSrM2RXrviFnwe7LmvEB5YVDGmm3mN5w" 
environment = "your_environment"  # Replace with your environment, e.g., "us-west1-gcp"

# Initialize Pinecone
pc = Pinecone(
    api_key=api_key,
    environment=environment
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
            region='us-west-2'  # Replace with your actual region
        )
    )

# Connect to the index
index = pc.index(index_name)

# Load vectorized data
import json
with open("vectorized_courses.json", "r") as file:
    vectorized_data = json.load(file)

# Upsert data into Pinecone
batch_size = 100
upserts = []
for record in vectorized_data:
    upserts.append({
        "id": record["course_name"],  # Unique ID
        "values": record["embedding"],  # Embedding vector
        "metadata": {"category": record["category"]}  # Metadata for filtering
    })

# Process in batches
for i in range(0, len(upserts), batch_size):
    index.upsert(vectors=upserts[i:i + batch_size])

print("Data successfully upserted into Pinecone!")
