import streamlit as st
import yt_dlp
import sys
import torch
from pytube import YouTube
from transformers import BartTokenizer, BartForConditionalGeneration
from reportlab.pdfgen import canvas
import tempfile
import openai
import requests
import os
import nltk
nltk.download('punkt')
# Set up OpenAI API key
openai.api_key = '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'

st.set_page_config(page_title="YouTube Video Summary", page_icon=":clapper:", layout="wide")

@st.cache_resource
def load_finetuned_bart_model():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    return tokenizer, model

def download_audio(youtube_link):
    download_message = st.empty()
    with st.spinner("Downloading video audio..."):
        download_message.text("Downloading video audio... This might take a few minutes.")
        # Create a temporary file to store the audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".%(ext)s")
        temp_file_name = temp_file.name
        temp_file.close()  # Close the file handle so yt-dlp can write to it

        # Configure yt-dlp options to download the best available audio in webm format
        ydl_opts = {
            'format': 'bestaudio',
            'outtmpl': temp_file_name,  # Template file name; '%(ext)s' will be replaced by the correct extension
            'quiet': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_link])
        download_message.text("Download completed!")

    # Replace the placeholder with the actual extension 'mp3'
    mp3_file = temp_file_name.replace("%(ext)s", "mp3")

    # Verify if the file was created
    if os.path.isfile(mp3_file):
        download_message.text("Download completed successfully!")
        print("File created:", mp3_file)
        return mp3_file
    else:
        download_message.text("Download failed: Audio file was not created.")
        print("Error: Audio file was not created.")
        return None

def transcribe_audio_deepgram(file_path):
    API_KEY = "b0c631b38640077a1e2530d3cf53f3e1e6a95e4a"

    # Deepgram transcription endpoint; you can specify the language (here: English US)
    url = "https://api.deepgram.com/v1/listen?language=en-US"

    # Set the authorization header using your API key
    headers = {
        "Authorization": f"Token {API_KEY}"
    }

    # Open your audio file (ensure it is within the allowed size/format limits)
    with open(file_path, "rb") as audio_file:
        # Post the audio data to the Deepgram endpoint
        response = requests.post(url, headers=headers, data=audio_file)

    # Check the response status and output the result
    if response.status_code == 200:
        transcription_result = response.json().get("results", {}).get("channels", [])[0].get("alternatives", [])[0].get("transcript", "")
        return transcription_result
    else:
        print("Error:", response.status_code)
        print(response.text)
        return {"error": response.text}


def split_text(text, max_words):
    """
    Splits text into chunks based on sentence boundaries using NLTK,
    ensuring that each chunk has approximately max_words.
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        # Count words in the sentence
        sentence_length = len(sentence.split())
        # If adding this sentence exceeds the max_words and we already have content, start a new chunk
        if current_length + sentence_length > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def generate_summary(text, tokenizer, model):
    """
    Generates a summary for the given text.
    The text is first split into manageable chunks,
    each chunk is summarized, and then the individual summaries
    are re-summarized to produce a final, more coherent output.
    """
    progress_bar = st.progress(0)
    status_message = st.empty()

    status_message.text("Generating detailed summary...")
    progress_bar.progress(10)

    # Define a maximum number of words per chunk (adjustable)
    max_chunk_words = 200
    text_chunks = split_text(text, max_chunk_words)
    summary_chunks = []
    num_chunks = len(text_chunks)

    try:
        # Summarize each chunk individually
        for i, chunk in enumerate(text_chunks):
            input_tokens = tokenizer.encode(chunk, return_tensors='pt', max_length=1024, truncation=True)
            summary_ids = model.generate(
                input_tokens,
                num_beams=8,
                length_penalty=1.0,
                max_length=300,    # Adjust as needed
                min_length=100,    # Adjust as needed
                no_repeat_ngram_size=2,
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )
            summary_text = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
            summary_chunks.append(summary_text)
            # Update progress based on the number of chunks processed
            progress_bar.progress(10 + int(((i+1)/num_chunks)*70))
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return "Error generating summary. Please try again."

    # Combine the individual summaries
    combined_summary = ' '.join(summary_chunks)
    status_message.text("Refining final summary...")
    progress_bar.progress(90)

    # Optionally re-summarize the concatenated summary for better coherence
    input_tokens = tokenizer.encode(combined_summary, return_tensors='pt', max_length=1024, truncation=True)
    final_summary_ids = model.generate(
        input_tokens,
        num_beams=8,
        length_penalty=1.0,
        max_length=350,    # Increased to allow a more natural ending
        min_length=150,    # Adjust as needed
        no_repeat_ngram_size=2,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    final_summary = tokenizer.decode(final_summary_ids.squeeze(), skip_special_tokens=True)

    progress_bar.progress(100)
    status_message.text("Summary generation completed!")
    return final_summary


def export_text(text, filename="export.txt"):
    with open(filename, "w") as file:
        file.write(text)
    return filename

def export_pdf(text, filename="export.pdf"):
    c = canvas.Canvas(filename)
    c.drawString(100, 800, text)
    c.save()
    return filename

def main():
    col1, col2 = st.columns([1, 9])
    with col1:
        st.image("yt.png", width=100)
    with col2:
        st.title("YouTube Video Summarizer using WhisperAi and BART")

    youtube_link = st.text_input("Enter YouTube Video Link:", help="Paste the URL of the YouTube video you want to summarize.")

    if youtube_link:
        st.video(youtube_link)

    if st.button("Get Transcript") and youtube_link:
        audio_file_path = download_audio(youtube_link)
        transcript = transcribe_audio_deepgram(audio_file_path)
        st.session_state['transcript'] = transcript
        st.session_state['generate_summary_clicked'] = False
        st.session_state['summary_in_progress'] = False

    if 'transcript' in st.session_state:
        st.subheader("Transcript")
        transcript_output = st.text_area("Transcript Output", value=st.session_state['transcript'], height=300, help="You can scroll or edit the text if needed.", key='transcript_output')
        
        if st.button("Generate Summary"):
            st.session_state['generate_summary_clicked'] = True
            tokenizer, model = load_finetuned_bart_model()
            custom_summary = generate_summary(st.session_state['transcript'], tokenizer, model)
            print("Summary:", custom_summary)
            st.session_state['custom_summary'] = custom_summary

    if st.session_state.get('generate_summary_clicked'):
        st.subheader("Custom Summary")
        custom_summary_output = st.text_area("Custom Summary Output", value=st.session_state.get('custom_summary', ''), height=200, help="Here's the detailed summary.", key='custom_summary_output')

        export_col1, export_col2 = st.columns(2)
        with export_col1:
            if st.button("Export Transcript as TXT"):
                txt_path = export_text(st.session_state['transcript'], "transcript.txt")
                st.download_button(label="Download Transcript as TXT", data=open(txt_path, "rb"), file_name="transcript.txt", mime="text/plain")

        with export_col2:
            if st.button("Export Summary as PDF"):
                pdf_path = export_pdf(st.session_state.get('custom_summary', ''), "summary.pdf")
                st.download_button(label="Download Summary as PDF", data=open(pdf_path, "rb"), file_name="summary.pdf", mime="application/pdf")

    st.markdown("---")
    st.markdown("Built with :heart: using Streamlit, WhisperAi, and BART")

if __name__ == "__main__":
    main()
