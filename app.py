import streamlit as st
import sys
import torch
from pytube import YouTube
from transformers import BartTokenizer, BartForConditionalGeneration
from reportlab.pdfgen import canvas
import tempfile
import openai

# Set up OpenAI API key
openai.api_key = ''

st.set_page_config(page_title="YouTube Video Summary", page_icon=":clapper:", layout="wide")

@st.cache_resource
def load_finetuned_bart_model():
    tokenizer = BartTokenizer.from_pretrained('./finetuned_bart_model')
    model = BartForConditionalGeneration.from_pretrained('./finetuned_bart_model')
    return tokenizer, model

def download_audio(youtube_link):
    download_message = st.empty()
    with st.spinner("Downloading video audio..."):
        download_message.text("Downloading video audio... This might take a few minutes.")
        yt = YouTube(youtube_link)
        audio_stream = yt.streams.filter(only_audio=True, file_extension='webm').first()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
        audio_stream.download(filename=temp_file.name)
        download_message.text("Download completed!")
    return temp_file.name

def transcribe_audio_openai(file_path):
    with open(file_path, "rb") as audio_file:
        transcription = openai.audio.translations.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    return transcription

def split_text(text, max_length):
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_length:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return chunks

def generate_summary(text, tokenizer, model):
    progress_bar = st.progress(0)
    status_message = st.empty()
    
    status_message.text("Generating detailed summary...")
    progress_bar.progress(25)
    
    max_chunk_length = 1024  # Use larger chunk length for better summaries
    text_chunks = split_text(text, max_chunk_length)
    summary_chunks = []
    
    try:
        for chunk in text_chunks:
            input_tokens = tokenizer.batch_encode_plus([chunk], return_tensors='pt', max_length=1024, truncation=True)['input_ids']
            summary_ids = model.generate(
                input_tokens, 
                num_beams=8,  # Increased for more diversity in generation
                length_penalty=1.0,  # Adjusted for balanced length
                max_length=300,  # Increased max length for more detailed summaries
                min_length=100,  # Ensure minimum length for substance
                no_repeat_ngram_size=2,  # Adjusted to allow some repetition for coherence
                early_stopping=True,
                do_sample=True,  # Enable sampling for more diverse output
                top_k=50,  # Sample from top 50 tokens
                top_p=0.95  # Use nucleus sampling
            )
            summary_text = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
            summary_chunks.append(summary_text)
        summary_text = ' '.join(summary_chunks)
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        summary_text = "Error generating summary. Please try again."
    
    progress_bar.progress(100)
    status_message.text("Summary generation completed!")
    
    return summary_text

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
        transcript = transcribe_audio_openai(audio_file_path)
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
