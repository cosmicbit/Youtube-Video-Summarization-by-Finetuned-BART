import streamlit as st
import sys
import torch
from pytube import YouTube
from transformers import BartTokenizer, BartForConditionalGeneration
from reportlab.pdfgen import canvas
import tempfile
import openai
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled, NoTranscriptAvailable

# Set up OpenAI API key
openai.api_key = ''

st.set_page_config(page_title="YouTube Video Summary", page_icon=":clapper:", layout="wide")

@st.cache_resource
def load_finetuned_bart_model():
    tokenizer = BartTokenizer.from_pretrained('./finetuned_bart_model')
    model = BartForConditionalGeneration.from_pretrained('./finetuned_bart_model')
    return tokenizer, model

# Function to get video ID from YouTube URL
def get_video_id(url):
    # Assuming the URL is in the format: https://www.youtube.com/watch?v=VIDEO_ID
    return url.split('v=')[-1]

# Function to extract transcript
def extract_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get manually created transcript
        transcript = transcript_list.find_transcript(['en'])
        return transcript.fetch(), 'en', True
    except NoTranscriptFound:
        # Try to get English auto-generated transcript
        try:
            transcript = transcript_list.find_generated_transcript(['en'])
            return transcript.fetch(), 'en', False
        except NoTranscriptFound:
            # Try to get Hindi auto-generated transcript
            try:
                transcript = transcript_list.find_generated_transcript(['hi'])
                return transcript.fetch(), 'hi', False
            except (NoTranscriptFound, TranscriptsDisabled, NoTranscriptAvailable):
                return None, None, None

# Function to split text into chunks
def split_text_into_chunks(text, max_length=3000):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(' '.join(current_chunk + [word])) <= max_length:
            current_chunk.append(word)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Function to translate and clean the transcript
def translate_and_clean_transcript(transcript_text, lang):
    # Split the transcript into manageable chunks
    chunks = split_text_into_chunks(transcript_text)
    
    processed_chunks = []
    for chunk in chunks:
        # Combine translation and correction in one prompt
        prompt = f"Please translate the following transcript from Hindi to English and correct any grammatical errors. If the transcript is already in English, just correct the errors.\n\n{chunk}"
        
        # Request OpenAI API to process the text
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates and corrects transcripts."},
                {"role": "user", "content": prompt}
            ]
        )
        
        processed_text = response.choices[0].message.content
        processed_chunks.append(processed_text)
    
    return ' '.join(processed_chunks)

# Function to clean the transcript
def clean_transcript(transcript_text):
    # Split the transcript into manageable chunks
    chunks = split_text_into_chunks(transcript_text)
    
    processed_chunks = []
    for chunk in chunks:
        # Correction prompt
        prompt = f"Please correct the following transcript for grammatical errors:\n\n{chunk}"
        
        # Request OpenAI API to process the text
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that corrects transcripts."},
                {"role": "user", "content": prompt}
            ]
        )
        
        processed_text = response.choices[0].message.content
        processed_chunks.append(processed_text)
    
    return ' '.join(processed_chunks)

# Function to process the transcript
def process_transcript(transcript, lang, is_manual):
    transcript_text = ' '.join([t['text'] for t in transcript])
    if is_manual:
        return transcript_text
    elif lang == 'en':
        return clean_transcript(transcript_text)
    else:
        return translate_and_clean_transcript(transcript_text, lang)

# Main function to get and process the transcript
def get_and_process_transcript(youtube_url):
    video_id = get_video_id(youtube_url)
    transcript, lang, is_manual = extract_transcript(video_id)
    
    if transcript:
        return process_transcript(transcript, lang, is_manual)
    else:
        return "No transcript or auto-generated captions available for this video."

# Function to split text
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

# Function to generate summary
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

# Function to export text
def export_text(text, filename="export.txt"):
    with open(filename, "w") as file:
        file.write(text)
    return filename

# Function to export pdf
def export_pdf(text, filename="export.pdf"):
    c = canvas.Canvas(filename)
    c.drawString(100, 800, text)
    c.save()
    return filename

# Main Streamlit app
def main():
    col1, col2 = st.columns([1, 9])
    with col1:
        st.image("yt.png", width=100)
    with col2:
        st.title("YouTube Video Summarizer using YouTubeTranscriptApi and BART")

    youtube_link = st.text_input("Enter YouTube Video Link:", help="Paste the URL of the YouTube video you want to summarize.")

    if youtube_link:
        st.video(youtube_link)

    if st.button("Get Transcript") and youtube_link:
        transcript = get_and_process_transcript(youtube_link)
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
    st.markdown("Built with :heart: using Streamlit, YouTubeTranscriptApi, and BART")

if __name__ == "__main__":
    main()
