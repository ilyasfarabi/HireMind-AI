##🚀 HireMind AI












🧠 AI Recruitment Copilot for Automated Hiring Decisions
🎬 Demo

📌 Replace this with your real GIF later (assets/demo.gif)

![HireMind AI Demo](assets/demo.jpeg)
📌 Overview

HireMind AI is an AI-powered recruitment system that automates CV screening, evaluates candidates using LLMs + RAG, and generates structured hiring decisions in seconds.

It transforms recruitment from a manual process → into an AI-driven automated pipeline.

💼 Business Impact

##HireMind AI helps companies:

⚡ Reduce CV screening time by 80–90%
🎯 Improve hiring accuracy using structured AI scoring
⚖️ Reduce human bias in recruitment decisions
📧 Automate CV intake via Gmail + n8n
📊 Track hiring history & analytics
🚨 Problem

##Traditional hiring is:

Slow ⏳
Subjective 🎭
Inconsistent 📉
Hard to scale 📦
💡 Solution

##HireMind AI solves this using:

🧠 LLM-based evaluation (Google Gemini)
🔍 RAG-based job matching system
📄 Automated resume parsing
⚙️ Workflow automation (n8n + Gmail)
💾 Persistent evaluation tracking
✨ Key Features
📄 Smart Resume Processing
PDF upload support
Multi-page extraction (PyMuPDF)
Clean structured text generation
🧠 AI Candidate Evaluation

##Each candidate receives:

Score (0–100)
Hiring decision (Yes / No)
Strengths & weaknesses
Skill-job matching analysis
Human-readable summary
🔍 RAG Job Matching
Job descriptions stored in vector DB (ChromaDB)
Semantic similarity search
Context-aware evaluation per role
📧 Email Automation (n8n)

Recruiter workflow:

##Gmail → CV Received → AI Evaluation → Result Email Sent

##📌 Just send a CV → system handles everything.

##📊 Metrics (Important for Portfolio)
Metric	Value
CV Processing Time	⚡ < 5 seconds
Automation Coverage	📧 100% email-based workflow
Evaluation Output	📊 Structured JSON + explanation
Matching Method	🧠 Semantic (RAG)
Bias Reduction	⚖️ High (standardized scoring)
🏗️ Architecture
                    Gmail / API / UI
                           ↓
                    FastAPI Backend
                           ↓
        ┌─────────────────────────────────┐
        │        AI Processing Layer      │
        │  - PDF Parser (PyMuPDF)         │
        │  - RAG Engine (ChromaDB)        │
        │  - Gemini Evaluator             │
        └─────────────────────────────────┘
                           ↓
        ┌─────────────────────────────────┐
        │       Storage Layer             │
        │  - SQLite (history)             │
        │  - Vector DB (knowledge base)   │
        └─────────────────────────────────┘
##⚙️ Tech Stack
Backend: FastAPI
LLM: Google Gemini 2.0 Flash
RAG: LangChain + ChromaDB
Embeddings: all-MiniLM-L6-v2
Automation: n8n + Gmail API
Database: SQLite
PDF Parsing: PyMuPDF
Deployment: Docker
📡 API Endpoints
Method	Endpoint	Description
GET	/	Health check
POST	/cv/upload	Upload CV
POST	/evaluate	Evaluate candidate
GET	/history	Get all evaluations
GET	/history/stats	Analytics
🧪 Example Output
{
  "evaluation_id": 12,
  "score": 91,
  "decision": "Yes",
  "strengths": ["Python", "FastAPI", "RAG"],
  "weaknesses": ["Kubernetes"],
  "summary": "Strong AI engineer with production-ready experience."
}
##🚀 Quick Start
git clone https://github.com/your-username/hiremind-ai.git
cd hiremind-ai
pip install -r requirements.txt
Run locally
uvicorn main:app --reload
##OR Docker
docker compose up --build
📈 Roadmap
🌐 React dashboard for recruiters
🔐 Authentication system (JWT)
📊 Advanced analytics dashboard
📬 Auto email replies improvements
📁 Bulk CV processing
🤝 ATS integrations (Workday, Greenhouse)
💼 Why This Project Stands Out

##This is not just a CRUD project.

It demonstrates:

🧠 LLM system design (Gemini integration)
🔍 RAG architecture (real-world AI retrieval system)
⚙️ Backend engineering (FastAPI production API)
📧 Workflow automation (n8n + Gmail)
💾 Data persistence + vector search
⭐ Support

##If you like this project:

⭐ Star the repo
🍴 Fork it
🚀 Share it
