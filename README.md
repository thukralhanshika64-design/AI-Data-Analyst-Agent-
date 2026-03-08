# 📊 AI Data Analyst Agent

An AI-powered data analysis web app that performs **automated Exploratory Data Analysis (EDA)** on any CSV dataset. Upload your data, and the AI returns structured insights — column analysis, statistics, anomalies, and actionable recommendations — all rendered in a premium dashboard.

![AI Data Analyst Agent](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white) ![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white) ![LangChain](https://img.shields.io/badge/LangChain-Powered-1C3C3C?logo=chainlink&logoColor=white)

## ✨ Features

- **📁 Drag-and-Drop CSV Upload** — Works with any dataset up to 200 MB
- **🤖 Dual AI Backends** — Ollama (local, private) or Groq (free cloud API)
- **📐 Full EDA Pipeline** — Summary stats, column analysis, distributions, anomalies
- **💡 Smart Insights** — AI-generated actionable business/analytical insights
- **⚠️ Anomaly Detection** — Flags nulls, outliers, duplicates, suspicious patterns
- **✅ Recommendations** — Specific, actionable next steps for your data
- **🎨 Premium Dashboard UI** — Gradient cards, styled tables, horizontal bar charts
- **🔒 Privacy-First** — Local mode keeps all data on your machine

## 🚀 Quick Start (Local)

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed with at least one model

### Setup

```bash
# Clone the repo
git clone https://github.com/thukralhanshika3-glitch/ai-data-analyst.git
cd ai-data-analyst

# Install dependencies
pip install -r requirements.txt

# Pull a model (if you haven't already)
ollama pull llama3

# Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**. Select "Ollama (Local)" in the sidebar and start analyzing!

## ☁️ Deploy to Streamlit Cloud (Free)

This app can be deployed for free on [Streamlit Community Cloud](https://streamlit.io/cloud) so **anyone with a link can use it**.

### Step 1: Get a Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free)
3. Go to **API Keys** → **Create API Key**
4. Copy the key

### Step 2: Deploy on Streamlit Cloud

1. Push this repo to GitHub (or fork it)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **"New app"**
5. Select your repo, branch `main`, and file `app.py`
6. Under **Advanced settings** → **Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
7. Click **Deploy!**

Your app will be live at `https://your-app-name.streamlit.app` — share the link with anyone! 🎉

## 🏗️ Project Structure

```
ai-data-analyst/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── sample_data.csv         # Sample dataset for testing
├── .gitignore
├── .streamlit/
│   └── config.toml         # Streamlit theme configuration
└── README.md
```

## 🔧 Configuration

### Supported AI Providers

| Provider | Models | Cost | Privacy |
|----------|--------|------|---------|
| **Ollama (Local)** | llama3, mistral, phi3, qwen2.5-coder | Free | ✅ 100% local |
| **Groq (Cloud)** | llama-3.3-70b, llama-3.1-8b, gemma2-9b, mixtral-8x7b | Free tier | ☁️ API call |

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Groq API key for cloud mode | Only for Groq |

## 📝 How It Works

1. **Upload** — User drops a CSV file
2. **Extract** — App reads the data and sends it to the LLM with a structured JSON prompt
3. **Analyze** — The AI returns a strict JSON object with summary, stats, insights, anomalies
4. **Render** — The dashboard parses the JSON and renders each section as a premium card/table

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

## 📄 License

MIT License — free to use, modify, and distribute.
