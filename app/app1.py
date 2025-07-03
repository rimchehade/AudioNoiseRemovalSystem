from flask import Flask, request, send_file, render_template
import os
import numpy as np
import torch
from torch.autograd import Variable
from scipy.io import wavfile
from tqdm import tqdm
import uuid
from pydub import AudioSegment
from model import Generator, emphasis, slice_signal

# Constants from your model
window_size = 2 ** 14  # 16384 samples
sample_rate = 16000



# Load the generator model once at startup
generator = Generator()
generator.load_state_dict(torch.load(r'epochs/generator-46.pkl', map_location='cpu'))  # Adjust epoch file as needed
if torch.cuda.is_available():
    generator.cuda()
generator.eval()

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Function to process audio
def process_audio(file_path):
    try:
        noisy_slices = slice_signal(file_path, window_size, 1, sample_rate)
        enhanced_speech = []
        for noisy_slice in tqdm(noisy_slices, desc='Generate enhanced audio'):
            z = torch.randn(1, 1024, 8)  # Updated to torch.randn
            noisy_slice = torch.from_numpy(emphasis(noisy_slice[np.newaxis, np.newaxis, :])).type(torch.FloatTensor)
            if torch.cuda.is_available():
                noisy_slice, z = noisy_slice.cuda(), z.cuda()
            noisy_slice, z = Variable(noisy_slice), Variable(z)
            with torch.no_grad():
                generated_speech = generator(noisy_slice, z).data.cpu().numpy()
            generated_speech = emphasis(generated_speech, emph_coeff=0.95, pre=False)
            generated_speech = generated_speech.reshape(-1)
            enhanced_speech.append(generated_speech)
        enhanced_speech = np.array(enhanced_speech).reshape(1, -1)
            # — convert to 16-bit PCM —
        flat = enhanced_speech.reshape(-1)
        mx = np.max(np.abs(flat))
        if mx > 0:
            flat = flat / mx
        int16_audio = (flat * 32767).astype(np.int16)

        output_file = os.path.join(
            OUTPUT_FOLDER,
            f"enhanced_{os.path.basename(file_path).split('.')[0]}.wav"
        )
        wavfile.write(output_file, sample_rate, int16_audio)
        return output_file
    except Exception as e:
        raise Exception(f"Error processing audio: {str(e)}")

# Routes
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part", 400
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        # Save uploaded file
        filename = f"{file.filename}"
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(input_path)

        # Simulate enhancement (replace this with your model)
        enhanced_filename = f"enhanced_{filename}"
        output_path = process_audio(input_path)
        enhanced_filename = os.path.basename(output_path)

        # Return download page
        return render_template("download.html", filename=enhanced_filename, original_filename=filename)

    return render_template("upload.html")

@app.route("/record", methods=["GET", "POST"])
def record_audio():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part", 400
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        filename = f"{file.filename}"
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(input_path)

        # Process audio with the model
        output_path = process_audio(input_path)
        enhanced_filename = os.path.basename(output_path)

        return render_template("download.html", filename=enhanced_filename, original_filename=filename)

    return render_template("record.html")

@app.route("/download/<filename>")
def download_file(filename):
    return send_file(os.path.join(app.config["OUTPUT_FOLDER"], filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)