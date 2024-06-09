# IntelliBlend: Chat and Seamlessly Integrate Knowledge from Websites, PDFs, and Videos for Smarter Conversations

IntelliBlend is an interactive application built using Streamlit that leverages LangChain, a Python package for language processing tasks, to facilitate intelligent conversations and integrate knowledge from diverse sources such as websites, PDFs, and YouTube videos. The application aims to enhance user interactions by providing relevant information and responses based on their queries.

## Features

- **Natural Language Chat:** Users can engage in conversations with the system using natural language.
- **Knowledge Retrieval:** The system can retrieve information from websites, PDFs, and YouTube videos to provide contextually relevant responses.
- **Similarity Scores:** IntelliBlend calculates similarity scores between user input and system-generated responses to assess the relevance of the answers.
- **BLEU Scores:** The application evaluates response generation using BLEU scores, which measure the similarity between the system's responses and reference answers (in this case, user inputs).
- **Sentiment Analysis:** IntelliBlend analyzes the sentiment of system responses to ensure that they are contextually appropriate.
- **Token Usage Monitoring:** The application monitors token usage for resource management, ensuring efficient processing of user queries.

## Technologies Used

- **LangChain:** LangChain is a Python package used for various language processing tasks, including generating responses and retrieving information.
- **Streamlit:** Streamlit is a web application framework used for building interactive web applications, providing a user-friendly interface for IntelliBlend.
- **PyPDF2:** PyPDF2 is a Python library used for reading PDF files, enabling the application to extract information from PDF documents.
- **NLTK:** NLTK is a Python library used for natural language processing tasks such as tokenization, stemming, and parsing, enhancing the language processing capabilities of IntelliBlend.
- **TextBlob:** TextBlob is a Python library used for processing textual data, including sentiment analysis, which helps IntelliBlend analyze the sentiment of its responses.
- **scikit-learn:** scikit-learn is a Python library used for various machine learning tasks, including calculating similarity scores between texts, which is essential for evaluating response relevance in IntelliBlend.
- **FAISS:** FAISS is a library used for efficient similarity search and clustering of dense vectors, enhancing the speed and accuracy of similarity calculations in IntelliBlend.
- **OpenAI:** OpenAI is a platform used for building and deploying AI models, which is leveraged by IntelliBlend for language modeling and response generation.
- **Chroma:** Chroma is a Python package used for managing vector stores and embeddings, which are essential for storing and processing textual data in IntelliBlend.

## How to Run

To run IntelliBlend, follow these steps:

1. Clone the repository: `git clone https://github.com/EvanVR/RAG_LLM_Application_Langchain`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Streamlit application: `streamlit run app.py`

## Usage

1. Enter a website URL, upload a PDF, or enter a YouTube video URL to provide context for the conversation.
2. Start a conversation with the system by typing messages in the chat box.
3. The system will respond based on the input and the knowledge it has from the provided sources.
4. View the similarity score, BLEU score, sentiment, and token usage for each conversation to assess the system's performance.

## Contributors

- Evan Velagaleti (ev379@drexel.edu)
