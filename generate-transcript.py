import whisper
from pyannote.audio import Pipeline
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import csv
import subprocess
import sys

def get_number_of_speakers():
    """
    Prompt the user to enter the number of speakers.
    Ensures that the input is a positive integer.
    """
    while True:
        try:
            num = int(input("Enter the expected number of speakers: "))
            if num < 1:
                raise ValueError("Number of speakers must be at least 1.")
            return num
        except ValueError as ve:
            print(f"Invalid input: {ve}. Please enter a positive integer.")

def convert_to_wav(input_file):
    """
    Converts the input audio file to WAV format using FFmpeg.
    Returns the path to the converted WAV file.
    """
    output_file = os.path.splitext(input_file)[0] + ".wav"
    if not os.path.exists(output_file):
        try:
            print(f"Converting {input_file} to WAV format...")
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_file, output_file],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"Conversion successful: {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting file: {e.stderr.decode('utf-8')}")
            sys.exit(1)
    else:
        print(f"WAV file already exists: {output_file}")
    return output_file

def sanitize_filename(file_path):
    """
    Replaces spaces in the filename with underscores to avoid RTTM format issues.
    Returns the sanitized file path.
    """
    directory, filename = os.path.split(file_path)
    sanitized_filename = filename.replace(" ", "_")
    sanitized_path = os.path.join(directory, sanitized_filename)
    if file_path != sanitized_path:
        try:
            os.rename(file_path, sanitized_path)
            print(f"Renamed file to: {sanitized_path}")
        except OSError as e:
            print(f"Error renaming file: {e}")
            sys.exit(1)
    else:
        print("Filename does not contain spaces. No renaming needed.")
    return sanitized_path

def select_file():
    """
    Opens a Finder dialog for the user to select an audio file.
    Returns the selected file path.
    """
    root = Tk()
    root.withdraw()  # Hide the root window
    root.call('wm', 'attributes', '.', '-topmost', True)  # Bring dialog to front
    file_path = askopenfilename(
        title="Select an Audio File",
        filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.flac")]
    )
    root.destroy()
    return file_path

def main():
    print("=== Audio Transcription and Speaker Diarization ===\n")

    # Step 1: Select Audio File
    file_path = select_file()
    if not file_path:
        print("No file selected. Exiting...")
        sys.exit(0)
    print(f"Selected file: {file_path}\n")

    # Step 2: Prompt for Number of Speakers
    num_speakers = get_number_of_speakers()
    print(f"Number of speakers set to: {num_speakers}\n")
   
   
    # Step 3: Sanitize Filename
    file_path = sanitize_filename(file_path)
    
    # Step 4: Convert to WAV if necessary
    file_path = convert_to_wav(file_path)
    
    # Verify the converted file exists
    if not os.path.exists(file_path):
        print(f"Converted file does not exist: {file_path}. Exiting...")
        sys.exit(1)
    
    # Step 5: Transcribe with Whisper
    print("Starting transcription with Whisper...")
    try:
        model = whisper.load_model("medium")
        result = model.transcribe(file_path)
        print(f"Transcription completed. Number of segments: {len(result['segments'])}\n")
    except Exception as e:
        print(f"Error during transcription: {e}")
        sys.exit(1)
    
    # Optional: Save Whisper transcription to a text file for verification
    transcription_text = "\n".join([segment['text'] for segment in result['segments']])
    transcription_output = os.path.splitext(file_path)[0] + "_transcription.txt"
    try:
        with open(transcription_output, "w", encoding='utf-8') as txt_file:
            txt_file.write(transcription_text)
        print(f"Whisper transcription saved to {transcription_output}\n")
    except Exception as e:
        print(f"Error saving transcription text file: {e}")
        sys.exit(1)
    
    # Step 6: Perform Speaker Diarization
    print("Starting speaker diarization with Pyannote...")
    try:
        # Initialize the pipeline with the specified number of speakers
        # Read token from file
        try:
            with open("token", "r") as f:
                hf_token = f.read().strip()
        except FileNotFoundError:
            print("Error: token file not found. Please create a 'token' file with your Hugging Face token.")
            sys.exit(1)
            
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1",
            use_auth_token=hf_token
        )
        diarization = pipeline(file_path, num_speakers=num_speakers)
        print(f"Diarization completed. Number of speaker segments: {len(diarization)}\n")
    except Exception as e:
        print("Failed to download or use the 'pyannote/speaker-diarization' model.")
        print("Ensure that you have accepted the model's terms on Hugging Face and that your token is correct.")
        print(f"Error details: {e}")
        sys.exit(1)
    
    # Step 7: Combine Transcription and Diarization
    print("Combining transcription and diarization results...")
    segments = []
    
    for idx, segment in enumerate(result['segments'], start=1):
        start = segment['start']
        end = segment['end']
        text = segment['text'].strip()
        
        # Find the speaker with the highest overlap with this transcription segment
        overlapping_speakers = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Check for overlap
            if (start < turn.end) and (end > turn.start):
                # Calculate overlap duration
                overlap_start = max(start, turn.start)
                overlap_end = min(end, turn.end)
                overlap = overlap_end - overlap_start
                overlapping_speakers.append((speaker, overlap))
        
        if overlapping_speakers:
            # Choose the speaker with the maximum overlap
            speaker = max(overlapping_speakers, key=lambda x: x[1])[0]
        else:
            speaker = "Unknown"
        
        segments.append({'speaker': speaker, 'start': start, 'end': end, 'text': text})
    
    print(f"Combined {len(segments)} transcription segments with diarization.\n")
    
    # Optional: Print sample segments for verification
    print("Sample combined segments:")
    for seg in segments[:5]:  # Print first 5 segments
        print(seg)
    print("\n")
    
    # Step 8: Save Results as CSV
    output_file = os.path.splitext(file_path)[0] + "_transcription_with_speakers.csv"
    try:
        with open(output_file, "w", newline="", encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Write header
            csvwriter.writerow(["Speaker", "Start Time (s)", "End Time (s)", "Text"])
            # Write rows
            for segment in segments:
                csvwriter.writerow([
                    segment['speaker'],
                    f"{segment['start']:.2f}",
                    f"{segment['end']:.2f}",
                    segment['text']
                ])
        print(f"CSV file saved successfully: {output_file}\n")
    except Exception as e:
        print(f"Error saving CSV file: {e}")
        sys.exit(1)
    
    print("=== Process Completed Successfully! ===")
    print(f"CSV file: {output_file}")
    print(f"Transcription Text file: {transcription_output}")

if __name__ == "__main__":
    main()