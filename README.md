
# Research Assistant with LangChain and LangServe

## Description

This research assistant, powered by LangChain and LangServe, streamlines research tasks by:

- Summarizing research papers and articles
- Generating insights and connections between different sources
- Answering questions based on provided research materials

## Installation

1. Clone the repository: `git clone https://github.com/your-username/research-assistant`
2. Install dependencies: `pip install -r requirements.txt`
**3. Set API Keys:**

   - **Gemini API Key:**
     - Create a `.env` file in the project's root directory.
     - Add the following line to the `.env` file, replacing `YOUR_GEMINI_API_KEY` with your actual key:

       ```
       GEMINI_API_KEY=YOUR_GEMINI_API_KEY
       ```

   - **LangChain API Key (for LangSmith tracing):**
     - Set the `LANGCHAIN_API_KEY` environment variable in your terminal or script before running the research assistant. For example:

       ```bash
       export LANGCHAIN_API_KEY=YOUR_LANGCHAIN_API_KEY
       ```

       Alternatively, you can add this line to your `.env` file:

       ```
       LANGCHAIN_API_KEY=YOUR_LANGCHAIN_API_KEY
       ```
## Usage

1. Run the main script: `python main.py`
2. Open your web browser and navigate to: `http://localhost:8000/research-assistant/playground/`
3. Conduct your research.


## Features

- **Summarization:** Summarizes research papers and articles concisely.
- **Insight Generation:** Identifies key points and relationships between sources.

## Contributing

Contributions are welcome! Please follow these guidelines:

- Submit pull requests with clear descriptions of changes.

## Acknowledgments

- Thanks to the LangChain team for their excellent libraries.