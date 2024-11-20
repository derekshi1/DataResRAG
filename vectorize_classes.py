import json
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')  # A small, efficient embedding model

with open("data_theory_req.json", "r") as file:
    data = json.load(file)

def get_embedding(course_name):
    return model.encode(course_name).tolist()  # Convert to Python list for JSON compatibility

# Recursive function to handle different levels of the JSON structure
def vectorize_courses(data):
    vectorized = {}
    for key, value in data.items():
        if isinstance(value, dict):  # If the value is another dictionary, recurse
            vectorized[key] = vectorize_courses(value)
        elif isinstance(value, list):  # If the value is a list, embed each course
            vectorized[key] = {course: get_embedding(course) for course in value}
    return vectorized

# Vectorize the data
vectorized_data = vectorize_courses(data)

# Save the vectorized data to a JSON file
with open("vectorized_courses.json", "w") as outfile:
    json.dump(vectorized_data, outfile)

print("Vectorization complete. Data saved to 'vectorized_courses.json'.")