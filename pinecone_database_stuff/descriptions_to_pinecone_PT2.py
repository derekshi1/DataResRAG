import os
from pinecone import Pinecone, ServerlessSpec
import json
import pandas as pd

# Initialize Pinecone with your API key
api_key = "pcsk_2jV2Ge_7K1scdaDQtdnBxtLte8hp7VczDzUAQWopd8ZgYP48Lx2Rm4zuppYg2qjT2WseGg" 
# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

# Create or connect to the index for descriptions
index_name = "course-descriptions-combined"
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

# Load your DataFrame with vectorized courses
df = pd.read_json("vectorized_combined_courses.jsonl", lines=True)
df["description"] = df["description"].fillna("Unknown")
df["category"] = df["category"].fillna("Unknown")
df["sequence"] = df["sequence"].fillna("None")

# Check embedding dimensionality
print(df["embedding"].apply(lambda x: len(x)).unique())  # Should print [384] if using MiniLM

# Prepare data for upsert
batch_size = 100
upserts = []

# Upsert vectorized courses in batches
for _, row in df.iterrows():
    try:
        # Ensure the necessary columns exist
        if "course_id" in row and "embedding" in row and "description" in row:
            upserts.append({
                "id": row["course_id"],  # Unique course ID
                "values": row["embedding"],  # Embedding vector
                "metadata": {
                    "description": row["description"],  # Course description
                    "category": row["category"],  # Course category
                    "sequence": row["sequence"],  # Sequence (if applicable)
                }
            })
        else:
            print(f"Missing keys in record: {row}")

        # Perform upserts in batches
        if len(upserts) == batch_size:
            print(f"Upserting {len(upserts)} records...")  # Debug: print the number of records being upserted
            index.upsert(vectors=upserts)
            print(f"Upserted {len(upserts)} records.")
            upserts = []  # Clear the batch

    except Exception as e:
        print(f"Error processing record {row['course_id']}: {e}")

# Final upsert for remaining records
if upserts:
    print(f"Upserting remaining {len(upserts)} records...")  # Debug: print before final upsert
    index.upsert(vectors=upserts)
    print(f"Upserted {len(upserts)} remaining records.")

print(f"Data successfully upserted into index '{index_name}'.")





'''
api_key = "pcsk_2jV2Ge_7K1scdaDQtdnBxtLte8hp7VczDzUAQWopd8ZgYP48Lx2Rm4zuppYg2qjT2WseGg"
# Initialize Pinecone
pc = Pinecone(api_key=api_key)

# Create or connect to the index for descriptions
index_name = "course-descriptions-combined"
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
    with open("vectorized_combined_courses.jsonl", "r") as file:
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



# Initialize Pinecone
pinecone.init(api_key="pcsk_2jV2Ge_7K1scdaDQtdnBxtLte8hp7VczDzUAQWopd8ZgYP48Lx2Rm4zuppYg2qjT2WseGg")
index_name = "course-planning"

# Create the index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384)

index = pinecone.Index(index_name)

# Prepare data for Pinecone upsert
def prepare_pinecone_data(df):
    records = []
    for _, row in df.iterrows():
        records.append({
            "id": row["course_id"],
            "values": row["embedding"],
            "metadata": {
                "description": row["description"],
                "category": row["category"],
                "sequence": row["sequence"]
            }
        })
    return records
df = pd.read_csv("vectorized_combined_courses.csv")
data_to_upsert = prepare_pinecone_data(df)

# Step 8: Upsert data to Pinecone
index.upsert(vectors=data_to_upsert)
print(f"Upserted {len(data_to_upsert)} records to Pinecone index '{index_name}'.")

'''
"""
# Step 9: Query the Pinecone index (example query)
query_text = "I want an introductory course on machine learning."
query_vector = model.encode(query_text)

results = index.query(vector=query_vector, top_k=5, include_metadata=True)

# Step 10: Display query results
print("Query Results:")
for match in results["matches"]:
    print(f"Course ID: {match['id']}")
    print(f"Description: {match['metadata']['description']}")
    print(f"Category: {match['metadata']['category']}")
    print(f"Score: {match['score']}")
    print()
"""