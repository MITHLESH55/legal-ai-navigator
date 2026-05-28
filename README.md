![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-Backend-black)
![Neo4j](https://img.shields.io/badge/Neo4j-GraphDB-green)
![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-red)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![LLM](https://img.shields.io/badge/AI-Gemini/Groq-purple)


🧠 Legal AI Navigator

An **AI-powered Legal Assistant** that helps users understand Indian legal concepts, IPC sections, court procedures, and case-related information using Large Language Models (LLMs), Knowledge Graphs, and Vector Search.

Built with **Groq / Gemini LLMs**, **Neo4j**, **Qdrant**, and **Docker**.

 Features

-  AI-powered legal question answering
-  Explanation of IPC sections with punishment
-  Context-aware conversations
-  PDF & document-based legal queries
-  Knowledge Graph using Neo4j
-  Semantic search using Qdrant
-  Fully Dockerized multi-container architecture
-  Secure environment variable handling


System Architecture

```text
                         ┌────────────────────┐
                         │      User UI       │
                         │   (index.html)     │
                         └─────────┬──────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │       Flask API          │
                    │         app.py           │
                    └──────────┬───────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
 ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
 │ Gemini / Groq  │  │ Neo4j Graph DB │  │ Qdrant Vector  │
 │ LLM Processing │  │ Knowledge Graph│  │ Semantic Search│
 └────────┬───────┘  └────────┬───────┘  └────────┬───────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │  Legal Reasoning Engine │
                 │    agent_logic.py       │
                 └──────────┬──────────────┘
                            │
                            ▼
                 ┌─────────────────────────┐
                 │ Response Generation     │
                 │ + Contextual Analysis   │
                 └─────────────────────────┘


📁 Project Structure

```text
legal-ai-navigator/
│
├── app.py
├── agent.py
├── agent_logic.py
├── supervisor_agent.py
├── tools.py
│
├── graph_builder.py
├── neo4j_queries.py
├── qdrant_store.py
│
├── browser_tool.py
├── drive_tool.py
├── court_tool.py
├── court_agents.py
│
├── dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
│
├── test_ecourts.py
├── test_hitl.py
├── test_document.pdf
│
└── index.html


 🔮 Future Enhancements

- AI-powered legal document summarization
- Real-time court case tracking
- Voice-based legal assistant
- Multi-language legal support
- Fine-tuned legal LLM integration
- Secure user authentication
- Legal analytics dashboard


## 👨‍💻 Author

Mithlesh Yadav  
BTech CSE | Symbiosis Institute of Technology

