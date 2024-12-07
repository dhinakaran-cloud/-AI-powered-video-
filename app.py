import os
from flask import Flask, request, render_template, jsonify
from moviepy.editor import VideoFileClip, concatenate_videoclips
import whisper
from google.cloud import vision
import openai

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize AI components
openai.api_key = "your_openai_api_key"
whisper_model = whisper.load_model("base")
vision_client = vision.ImageAnnotatorClient()

def transcribe_video(video_path):
    """Transcribes audio from a video file."""
    audio_path = "temp_audio.wav"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)
    result = whisper_model.transcribe(audio_path)
    os.remove(audio_path)
    return result['text']

def tag_video(video_path):
    """Extracts video frames and tags them using Google Vision API."""
    tags = []
    clip = VideoFileClip(video_path)
    for frame in clip.iter_frames(fps=1):  # Analyze one frame per second
        image = vision.Image(content=frame.tobytes())
        response = vision_client.label_detection(image=image)
        tags.extend([label.description for label in response.label_annotations])
    return list(set(tags))  # Remove duplicates

def generate_story(prompt, transcripts, tags):
    """Generates a cohesive story using GPT based on prompts, transcripts, and tags."""
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Create a story based on the following data:\nPrompt: {prompt}\nTranscripts: {transcripts}\nTags: {tags}",
        max_tokens=300,
    )
    return response.choices[0].text.strip()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle file uploads
        files = request.files.getlist("videos")
        prompt = request.form.get("prompt")
        transcripts = []
        tags = []

        for file in files:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Process videos
            transcripts.append(transcribe_video(file_path))
            tags.extend(tag_video(file_path))

        # Generate story
        story = generate_story(prompt, " ".join(transcripts), tags)

        return jsonify({"story": story, "tags": tags, "transcripts": transcripts})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
