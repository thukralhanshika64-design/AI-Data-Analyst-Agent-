import re
import torch
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

class VideoSummarizerLogic:
    def __init__(self, model_name="google/flan-t5-small"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

    @st.cache_resource
    def load_model(_self, model_name):
        """Loads and caches the tokenizer and model."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Use simple CPU loading since CUDA is False
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(_self.device)
        return tokenizer, model

    def extract_video_id(self, url):
        """Extracts video ID from different YouTube URL formats."""
        patterns = [
            r"(?:v=|youtu\.be/|embed/|v/|.+\?v=)([a-zA-Z0-9_-]{11})",
            r"(?:shorts/)([a-zA-Z0-9_-]{11})"
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def get_transcript(self, video_id):
        """Fetch transcript using the newer API format."""
        try:
            api = YouTubeTranscriptApi()
            # The .fetch method grabs the subtitle object list
            transcript = api.fetch(video_id)
            # We join the list into a single long string of text
            return " ".join([t.text for t in transcript])
        except TranscriptsDisabled:
            return "Error: Transcripts are disabled for this video."
        except NoTranscriptFound:
            return "Error: No transcript found for this video."
        except Exception as e:
            return f"Error: {str(e)}"

    def chunk_text(self, text, chunk_size=1200):
        """Splits transcript into manageable chunks (sliding window)."""
        sentences = text.split(". ")
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def summarize_chunk(self, text_chunk, tokenizer, model):
        """Summarizes a single chunk of text with optimized speed."""
        prompt = f"Summarize the following text briefly into one concise bullet point:\n{text_chunk}"
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512 # Reduced for speed
        ).to(self.device)

        with torch.no_grad():
            summary_ids = model.generate(
                **inputs,
                max_new_tokens=60, # Reduced for speed
                num_beams=1,       # Greedy search (fastest)
                early_stopping=True
            )

        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def refine_summary(self, notes_list, tokenizer, model):
        """Synthesizes fragment notes into a cohesive, high-insight summary."""
        combined_text = " ".join(notes_list)
        # Limit combined text to ensure it fits the context for the final pass
        prompt = f"Analyze these raw notes and provide a structured summary with 'Core Conclusion' and 'Top 3 Actionable Insights':\n{combined_text[:2000]}"
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        with torch.no_grad():
            summary_ids = model.generate(
                **inputs,
                max_new_tokens=250,
                num_beams=4, # Use better quality for the final synthesis
                early_stopping=True
            )

        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def process_video(self, url, model_name="google/flan-t5-small", progress_callback=None):
        """Main pipeline to process video with parallel synthesis and a final revision pass."""
        video_id = self.extract_video_id(url)
        if not video_id:
            return None, None, "Invalid YouTube URL."

        if progress_callback:
            progress_callback(0.1, "Fetching transcript...")
            
        transcript = self.get_transcript(video_id)
        if transcript.startswith("Error"):
            return None, None, transcript

        if progress_callback:
            progress_callback(0.2, "Chunking content...")
            
        chunks = self.chunk_text(transcript)
        
        if progress_callback:
            progress_callback(0.4, f"Powering up AI Engine ({model_name})...")
            
        tokenizer, model = self.load_model(model_name)
        
        if progress_callback:
            progress_callback(0.6, "Synthesizing segments in parallel...")

        # Step 1: Fragmented summarization
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(self.summarize_chunk, chunk, tokenizer, model) for chunk in chunks]
            fragmented_notes = [future.result() for future in futures]
            
        if progress_callback:
            progress_callback(0.85, "Revising notes for perfect insight...")

        # Step 2: Master Synthesis (Revision)
        master_insight = self.refine_summary(fragmented_notes, tokenizer, model)

        if progress_callback:
            progress_callback(1.0, "Analysis complete!")
            
        return fragmented_notes, master_insight, None
