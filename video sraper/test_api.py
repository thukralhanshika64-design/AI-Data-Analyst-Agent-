from youtube_transcript_api import YouTubeTranscriptApi
api = YouTubeTranscriptApi()
try:
    transcript = api.fetch("KLfer0MES2w")
    print(f"Type of transcript: {type(transcript)}")
    print(f"First element: {transcript[0]}")
    print(f"First element attributes: {dir(transcript[0])}")
    print("Successfully fetched transcript!")
except Exception as e:
    print(f"Error: {e}")
