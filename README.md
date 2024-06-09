# IntelliBlend: Chat and Seamlessly Integrate Knowledge from Websites, PDFs, and Videos for Smarter Conversations

IntelliBlend is a Streamlit-based application that allows users to chat with the system and seamlessly integrate knowledge from various sources such as websites, PDFs, and YouTube videos. The application uses LangChain, a Python package for language processing tasks, to enable intelligent conversations and provide relevant information based on user queries.

## Features
- Chat with the system using natural language.
- Retrieve information from websites, PDFs, and YouTube videos.
- Calculate similarity scores between user input and system responses.
- Calculate BLEU scores for response generation evaluation.
- Analyze sentiment of system responses.
- Monitor token usage for resource management.

## Technologies Used
- **LangChain**: A Python package for language processing tasks.
- **Streamlit**: A web application framework for building interactive web applications.
- **PyPDF2**: A Python library for reading PDF files.
- **NLTK**: A Python library for natural language processing tasks.
- **TextBlob**: A Python library for processing textual data.
- **scikit-learn**: A Python library for machine learning tasks.
- **FAISS**: A library for efficient similarity search and clustering of dense vectors.
- **OpenAI**: A platform for building and deploying AI models.
- **Chroma**: A Python package for vector stores and embeddings.

## How to Run
1. Clone the repository: `git clone https://github.com/EvanVR/RAG_LLM_Application_Langchain`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Streamlit application: `streamlit run app.py`

## Usage
1. Enter a website URL, upload a PDF, or enter a YouTube video URL.
2. Start a conversation with the system by typing messages in the chat box.
3. The system will respond based on the input and the knowledge it has from the provided sources.
4. View the similarity score, BLEU score, sentiment, and token usage for each conversation.

## Contributors
- Evan Velagaleti <ev379@drexel.edu>
