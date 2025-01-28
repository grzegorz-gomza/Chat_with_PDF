# Chat with Multiple PDFs

This project implements a Streamlit application that allows users to chat with multiple PDF documents. It extracts text from uploaded PDFs, processes the text into chunks, and uses OpenAI's model to provide conversational responses based on the document content.

## Features
- **PDF Upload**: Users can upload one or more PDF files for text extraction.
- **Interactive Chat**: Users can ask questions about the documents, and the application will respond with relevant information.
- **Memory Management**: The conversation history is maintained to provide context in ongoing discussions.

## Requirements
- Python 3.x
- Streamlit
- dotenv
- PyPDF2
- langchain and necessary submodules
- OpenAI API key

## Installation
1. Clone the repository.
2. Install the required packages using:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Run the application:
   ```
   streamlit run app.py
   ```

## Usage
1. Visit the application in your browser.
2. Upload PDF files in the sidebar.
3. Type your questions in the input box to interact with the content of the PDFs.

## Contributing
Feel free to submit issues or pull requests to improve functionality or documentation.

## License
This project is licensed under the MIT License.