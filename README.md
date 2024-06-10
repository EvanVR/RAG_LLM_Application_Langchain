# IntelliBlend: Chat and Seamlessly Integrate Knowledge from Websites, PDFs, and Videos for Smarter Conversations

- Evan Velagaleti (ev379@drexel.edu)

IntelliBlend is an interactive application built using Streamlit that leverages LangChain, a Python package for language processing tasks, to facilitate intelligent conversations and integrate knowledge from diverse sources such as websites, PDFs, and YouTube videos. The application aims to enhance user interactions by providing relevant information and responses based on their queries and the input source selected. This gives a blended result from every type of information source.

## Features

- **Natural Language Chat:** Users can engage in conversations with the system using natural language.
- **Knowledge Retrieval:** The system can retrieve information from websites, PDFs, and YouTube videos to provide contextually relevant responses.
- **Similarity Scores:** Similarity scores between user input with the information source and system-generated responses to assess the relevance of the answers.
- **BLEU Scores:** Evaluates response generation using BLEU scores, which measure the similarity between the system's responses and user inputs.
- **Sentiment Analysis:** IntelliBlend analyzes the sentiment of system responses to ensure that they are contextually appropriate.
- **Token Usage Monitoring:** The application monitors token usage for resource management, ensuring efficient processing of user queries.

## Technologies Used

- **LangChain:** for language processing tasks, generating responses, **Streamlit:** web application framework, **PyPDF2:** for reading PDF files, **NLTK:** for natural language processing tasks such as tokenization, stemming, and parsing,
- **TextBlob:**, **scikit-learn:** calculating similarity scores between texts, **FAISS:** clustering of dense vectors, **OpenAI:**  for language modeling and response generation, **Chroma:** for managing vector stores and embeddings, which are essential for storing and processing textual data.

## How to Run

To run IntelliBlend, follow these steps:

1. Clone the repository: `git clone https://github.com/EvanVR/RAG_LLM_Application_Langchain`
2. Install dependencies: `pip install -r requirements.txt`
3. Set OPENAI_API_KEY in the .env file or as a variable. Find your key: https://platform.openai.com/
4. Run the Streamlit application: `streamlit run app.py`

## Usage

1. Enter a website URL, upload a PDF, or enter a YouTube video URL to provide context for the conversation.
2. After pressing enter, after a few seconds, the user can type the prompt based on the information requested. 
3. Start a conversation with the system by typing messages in the chat box.
4. The system will respond based on the input and the knowledge it has from the provided sources.
5. View the similarity score, BLEU score, sentiment, and token usage for each conversation to assess the system's performance.
6. The session switching isn't managed for now, so if a new source needs to be selected, the application tab needs to be refreshed to interact with the new input source.


