# AI Video Summarizer: High-Performance Note Architect

A professional tool that leverages Large Language Models (LLMs) to transform long YouTube videos into concise, clear, and actionable notes. Built with Python, Google FLAN-T5, and Streamlit.

![Aesthetics](https://img.shields.io/badge/Aesthetics-Premium-blueviolet)
![Engine](https://img.shields.io/badge/Engine-FLAN--T5--Base-blue)
![Frontend](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B)

## ✨ Features

- **Automated Transcription**: Seamlessly pulls subtitles from any YouTube video.
- **Context-Aware Chunking**: Intelligent text splitting to fit the model's token limit.
- **State-of-the-Art Summarization**: Uses Google's FLAN-T5 model fine-tuned for instruction following.
- **Premium UI**: Modern dark-mode interface with glassmorphism and real-time progress.
- **Export Ready**: Download your summaries as Markdown files immediately.

## 🛠️ Installation

Ensure you have [uv](https://github.com/astral-sh/uv) installed.

```bash
# Clone the repository
git clone https://github.com/your-repo/video-summarizer.git
cd video-summarizer

# Install dependencies
uv sync

# Run the app
uv run streamlit run app.py
```

## 🧠 Technical Details

- **Model**: `google/flan-t5-base`
- **Tokenizer**: SentencePiece
- **Inference**: PyTorch (CUDA supported)
- **Transcription**: `youtube-transcript-api`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---
*Created by Antigravity AI*
