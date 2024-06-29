import streamlit as st
import torch
from pytube import YouTube
from transformers import pipeline
from reportlab.pdfgen import canvas
import openai
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled, NoTranscriptAvailable
import re

# Set up OpenAI API key
openai.api_key = ''

# Configure the Streamlit page with a title, icon, and wide layout
st.set_page_config(page_title="YouTube Video Summary", page_icon=":clapper:", layout="wide")

@st.cache_resource
def load_finetuned_bart_pipeline():
    """
    Load the fine-tuned BART model for summarization.
    Returns the summarization pipeline.
    """
    try:
        # Load the summarization pipeline using the fine-tuned BART model
        summarization_pipeline = pipeline(
            "summarization",
            model="./finetuned_bart_model",
            tokenizer="./finetuned_bart_model",
            device=0 if torch.cuda.is_available() else -1  # Use GPU if available
        )
        return summarization_pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_video_id(url):
    """
    Extract the video ID from a YouTube URL.
    """
    # Extract the video ID from the YouTube URL
    return url.split('v=')[-1]

def extract_transcript(video_id):
    """
    Extract the transcript from a YouTube video.
    Tries to get the manually created transcript first, then auto-generated English, and finally auto-generated Hindi transcript.
    """
    try:
        # List available transcripts for the video
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

def split_text_into_chunks(text, max_length=3000):
    """
    Split text into chunks of a specified maximum length.
    """
    words = text.split()
    chunks = []
    current_chunk = []

    # Split the text into chunks of specified max length
    for word in words:
        if len(' '.join(current_chunk + [word])) <= max_length:
            current_chunk.append(word)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def translate_and_clean_transcript(transcript_text, lang):
    """
    Translate and clean the transcript by correcting grammatical errors.
    """
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
                {"role": "system", "content": "You are a helpful assistant that translates the Hindi transcript into an English transcript and corrects the transcript of any grammatical errors and provides a clean grammatically correct English transcript."},
                {"role": "user", "content": chunk}
            ]
        )
        
        processed_text = response.choices[0].message.content
        processed_chunks.append(processed_text)
    
    return ' '.join(processed_chunks)

def clean_transcript(transcript_text):
    """
    Clean the transcript by correcting grammatical errors.
    """
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
                {"role": "user", "content": chunk}
            ]
        )
        
        processed_text = response.choices[0].message.content
        processed_chunks.append(processed_text)
    
    return ' '.join(processed_chunks)

def process_transcript(transcript, lang, is_manual):
    """
    Process the transcript: clean it if it's manual or English, translate and clean if it's in Hindi.
    """
    # Join the transcript text
    transcript_text = ' '.join([t['text'] for t in transcript])
    if is_manual:
        return transcript_text
    elif lang == 'en':
        return clean_transcript(transcript_text)
    else:
        return translate_and_clean_transcript(transcript_text, lang)

def get_and_process_transcript(youtube_url):
    """
    Extract and process the transcript from a YouTube video URL.
    """
    video_id = get_video_id(youtube_url)
    transcript, lang, is_manual = extract_transcript(video_id)
    if transcript:
        cleaned_transcript = process_transcript(transcript, lang, is_manual)
        return cleaned_transcript
    else:
        return "No transcript or auto-generated captions available for this video."

def split_text(text, chunk_size=600):
    """
    Split text into chunks of a specified size.
    """
    words = text.split()
    # Split the text into chunks of specified size
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def generate_summary(text, summarization_pipeline):
    """
    Generate a summary for the given text using the summarization pipeline.
    """
    progress_bar = st.progress(0)  # Display a progress bar
    status_message = st.empty()  # Placeholder for status messages

    status_message.text("Generating detailed summary...")  # Display status message
    progress_bar.progress(25)  # Update progress bar

    chunk_size = 600  # Set a slightly larger chunk size for better context
    text_chunks = split_text(text, chunk_size)
    summary_chunks = []

    if summarization_pipeline:
        try:
            for i, chunk in enumerate(text_chunks):
                st.write(f"Processing chunk {i+1}/{len(text_chunks)}: {chunk[:100]}...")  # Debugging: show the beginning of each chunk
                if len(chunk.split()) > chunk_size:
                    st.error(f"Chunk {i+1} is too long for the model to process.")
                    continue
                with torch.cuda.amp.autocast():
                    summary = summarization_pipeline(chunk, max_length=200, min_length=60, num_beams=2, do_sample=False)[0]['summary_text']
                summary_chunks.append(summary)
                progress_bar.progress(25 + int((i + 1) / len(text_chunks) * 75))  # Update progress bar
            summary_text = ' '.join(summary_chunks)
        except Exception as e:
            st.error(f"Error generating summary: {e}")
            summary_text = "Error generating summary. Please try again."
    else:
        summary_text = "Error loading summarization model."

    progress_bar.progress(100)  # Update progress bar
    status_message.text("Summary generation completed!")  # Update status message

    return summary_text

def export_text(text, filename="export.txt"):
    """
    Export the given text to a .txt file.
    """
    with open(filename, "w") as file:
        file.write(text)
    return filename

def export_pdf(text, filename="export.pdf"):
    """
    Export the given text to a .pdf file.
    """
    c = canvas.Canvas(filename)
    c.drawString(100, 800, text)
    c.save()
    return filename

def main():
    """
    Main function to run the Streamlit app.
    """
    col1, col2 = st.columns([1, 9])
    with col1:
        st.image("yt.png", width=100)
    with col2:
        st.title("YouTube Video Summarizer using BART")

    # Input field for YouTube video link
    youtube_link = st.text_input("Enter YouTube Video Link:", help="Paste the URL of the YouTube video you want to summarize.")

    if youtube_link:
        st.video(youtube_link)  # Display the video

    if st.button("Get Transcript") and youtube_link:
        # Extract and process transcript on button click
        with st.spinner('Extracting transcript...'):
            transcript = get_and_process_transcript(youtube_link)
        st.session_state['transcript'] = transcript
        st.session_state['generate_summary_clicked'] = False
        st.session_state['summary_in_progress'] = False

    if 'transcript' in st.session_state:
        st.subheader("Transcript")
        # Display the transcript in a text area
        transcript_output = st.text_area("Transcript Output", value=st.session_state['transcript'], height=300, help="You can scroll or edit the text if needed.", key='transcript_output')

        if st.button("Generate Summary"):
            # Generate summary on button click
            st.session_state['generate_summary_clicked'] = True
            summarization_pipeline = load_finetuned_bart_pipeline()
            custom_summary = generate_summary(st.session_state['transcript'], summarization_pipeline)
            st.session_state['custom_summary'] = custom_summary

    if st.session_state.get('generate_summary_clicked'):
        st.subheader("Custom Summary")
        # Display the custom summary in a text area
        custom_summary_output = st.text_area("Custom Summary Output", value=st.session_state.get('custom_summary', ''), height=200, help="Here's the detailed summary.", key='custom_summary_output')

        export_col1, export_col2 = st.columns(2)
        with export_col1:
            # Export the transcript as a TXT file
            if st.button("Export Transcript as TXT"):
                txt_path = export_text(st.session_state['transcript'], "transcript.txt")
                st.download_button(label="Download Transcript as TXT", data=open(txt_path, "rb").read(), file_name="transcript.txt", mime="text/plain")

        with export_col2:
            # Export the summary as a PDF file
            if st.button("Export Summary as PDF"):
                pdf_path = export_pdf(st.session_state.get('custom_summary', ''), "summary.pdf")
                st.download_button(label="Download Summary as PDF", data=open(pdf_path, "rb").read(), file_name="summary.pdf", mime="application/pdf")

    st.markdown("---")
    st.markdown("Built using Streamlit, YouTubeTranscriptApi, and BART")

if __name__ == "__main__":
    main()
