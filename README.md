# 🤖 Enterprise AI Copilot

An AI-powered Enterprise Knowledge Assistant built using **FastAPI, Streamlit, ChromaDB, and Ollama**.

Upload company documents, ingest websites, and chat with your organization's knowledge using Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

✅ Website Knowledge Ingestion

✅ PDF Knowledge Ingestion

✅ ChromaDB Vector Database

✅ Ollama Llama 3 Integration

✅ FastAPI Backend

✅ Streamlit Frontend

✅ Voice Input Support

✅ Multi-Company Knowledge Base

✅ Conversation History

✅ Enterprise-Style UI

---

## 📸 Demo

Add screenshots here.

### Dashboard

![Dashboard](screenshots/dashboard.png)

### Chat Interface

![Chat](screenshots/chat.png)

---

## 🏗️ Architecture

```text
Streamlit UI
      │
      ▼
FastAPI Backend
      │
      ▼
Website / PDF Ingestion
      │
      ▼
ChromaDB Vector Store
      │
      ▼
Ollama (Llama3)
      │
      ▼
AI Answers
```

---

## 🛠️ Tech Stack

### Backend

* FastAPI
* Python

### Frontend

* Streamlit

### AI / RAG

* ChromaDB
* Ollama
* Llama 3

### Document Processing

* BeautifulSoup
* pypdf

### Database

* SQLite

---

## 📂 Project Structure

```text
company-ai-copilot/
│
├── backend/
│   ├── main.py
│   ├── rag.py
│   ├── ingest.py
│   ├── scraper.py
│   ├── auth.py
│   └── requirements.txt
│
├── frontend/
│   └── app.py
│
├── chroma_db/
├── data/
└── README.md
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/enterprise-ai-copilot.git

cd enterprise-ai-copilot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
cd backend

pip install -r requirements.txt
```

---

## 🦙 Install Ollama

Download:

https://ollama.com

Pull Llama 3:

```bash
ollama pull llama3
```

Verify:

```bash
ollama list
```

---

## ▶️ Run Backend

Open Terminal 1

```bash
cd backend

uvicorn main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

---

## ▶️ Run Frontend

Open Terminal 2

```bash
cd frontend

streamlit run app.py
```

Frontend:

```text
http://localhost:8501
```

---

## 🧠 Example Workflow

### Step 1

Enter company name

```text
Tesla
```

### Step 2

Ingest website

```text
https://tesla.com
```

### Step 3

Ask questions

```text
What products does Tesla offer?
```

```text
Who is Tesla's CEO?
```

```text
What is Tesla's mission?
```

---

## 🔮 Future Improvements

* User Registration
* PostgreSQL Support
* Streaming Responses
* Analytics Dashboard
* OCR for Scanned PDFs
* Slack Integration
* WhatsApp Integration
* Multi-Language Support

---

## 🤝 Contributing

Contributions are welcome.

If you find bugs or have ideas for improvements:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

---

## ⭐ Support

If you found this project useful:

⭐ Star the repository

🍴 Fork it

🛠️ Contribute to it

---

## 👩‍💻 Author

**Priyanshi Kumrawat**

Final Year AI/ML Engineer

GitHub: https://github.com/Priyanshi-cell

LinkedIn: Add your LinkedIn profile here.
