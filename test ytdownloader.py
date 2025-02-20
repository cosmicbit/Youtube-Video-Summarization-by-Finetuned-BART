import yt_dlp

# URL of the video to download
video_url = "https://www.youtube.com/watch?v=aircAruvnKk"

# yt-dlp options
ydl_opts = {
    'format': 'best',  # Download the best available quality
    'outtmpl': 'test_video.%(ext)s',  # Save as test_video.mp4 or similar
}

# Download the video
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([video_url])

print("Download complete!")
