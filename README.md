# UCLA Course Planning Assistant using Retrieval-Augmented Generation (RAG)
## Project Overview
    This project leverages RAG to enhance course planning for math majors by providing personalized class recommendations. By combining user inputs with detailed course descriptions and utilizing advanced embeddings, we aim to deliver tailored academic advice.

## Workflow and Components
### 1. User Input
    A major plan worksheet with two lists of classes: required courses and elective courses (both lists separated by commas).
    Student's class preferences or academic interests.
### 2. Data Used to Enhance Output
    Course Descriptions: A dataset containing course codes and one-sentence descriptions of each course.
### 3. Embedding Process
    Formatting User Input:
      Extract and structure the lists of required and elective classes.
      Parse the user's one-sentence summary of interests.
    Embedding User Input:
      Convert the formatted user input into vector embeddings using a pre-trained language model.
    Formatting Course Data:
      Structure the course data to pair each course code with its corresponding description.
    Embedding Course Descriptions:
      Generate vector embeddings for each course description using the same language model.
### 4. Vector Database
    Creating a Vector Database:
      Use Pinecone to store the course embeddings in a highly scalable and efficient manner.
### 5. Similarity Search
    Performing a Similarity Search:
      Conduct a similarity search in the vector database to match user input embeddings with the most relevant course embeddings.
      Return a list of recommended courses based on the search results.
## Technologies Used
    Programming Languages: Python
    Machine Learning Libraries: Embeddings and natural language processing tools (e.g., Transformers)
    Vector Database: Pinecone
    RAG Framework: Leveraging state-of-the-art language models for embeddings and similarity search
