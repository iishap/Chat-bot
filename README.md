# CDP How-To Chatbot

A chatbot application built using **Streamlit**, **CrewAI**, **Groq API** & **LangChain**  to provide how-to guidance for major **Customer Data Platforms (CDPs)** like **Segment**, **mParticle**, **Lytics**, and **Zeotap**.

## Features
- Ask how-to questions about popular CDP platforms.
- Retrieve information from CDP documentation automatically.
- Web search fallback for additional information.
- Real-time chat interface with step-by-step responses.
- Multi-agent collaboration using **CrewAI**.
- Vector search with **FAISS** and **HuggingFace Embeddings**.

## Tech Stack
- **Python**
- **Streamlit** 
- **Groq API** 
- **CrewAI** 
- **LangChain**
- **FAISS** 
- **HuggingFace Embeddings**
- **DuckDuckGoSearch API** 

## Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd chatbot
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file and add your **Groq API key**.
```env
GROQ_API_KEY=your_api_key_here
```

4. Run the application:
```bash
streamlit run app.py
```

## How It Works

1. The chatbot takes user input questions about CDP platforms.
2. It identifies the relevant CDP from the question using **Regex Matching**.
3. Agents are initialized for:
   - Document Retrieval from CDP Docs (via Web Scraping)
   - Web Search using DuckDuckGo
   - Answer Generation
4. Retrieved data is indexed into **FAISS Vector Store** with sentence embeddings.
5. The answer generation agent combines information from docs and web search to generate a detailed step-by-step response.



## API Keys Required
- **Groq API Key** (for LLM responses)
- Optional: **DuckDuckGo API Key** (if web search is enabled)

## Contributing
Feel free to raise issues or create PRs to improve the functionality or add more CDPs.

## License
This project is licensed under the MIT License.

