import whisper
from pyannote.audio import Pipeline
import torch
import torchaudio
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import subprocess
import sys
from huggingface_hub import HfApi
import warnings
warnings.filterwarnings("ignore")

# Enable optimizations for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def get_hf_token():
    """
    Return the Hugging Face token from the token file.
    """
    try:
        with open("token", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print("Error: token file not found. Please create a 'token' file with your Hugging Face token.")
        sys.exit(1)


def verify_hf_token(token):
    """
    Verifies the given Hugging Face token using HfApi.
    Returns True if valid, otherwise False.
    """
    api = HfApi()
    try:
        user_info = api.whoami(token=token)
        if 'name' in user_info:
            return True
        return False
    except Exception:
        return False


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


def preprocess_audio(file_path):
    """
    Preprocess audio for better speaker diarization.
    Normalizes audio and ensures proper sample rate.
    """
    print("Preprocessing audio for better speaker recognition...")
    
    # Load audio
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        print("Converted stereo to mono")
    
    # Normalize audio
    waveform = waveform / torch.max(torch.abs(waveform))
    
    # Resample to 16kHz if needed (optimal for speaker diarization)
    if sample_rate != 16000:
        print(f"Resampling from {sample_rate}Hz to 16000Hz")
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_rate = 16000
    
    # Save preprocessed audio
    preprocessed_path = os.path.splitext(file_path)[0] + "_preprocessed.wav"
    torchaudio.save(preprocessed_path, waveform, sample_rate)
    print(f"Preprocessed audio saved to: {preprocessed_path}")
    
    return preprocessed_path


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
                ["ffmpeg", "-y", "-i", input_file, "-ac", "1", "-ar", "16000", output_file],
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
    Opens a file dialog for the user to select an audio file.
    Returns the selected file path.
    """
    root = Tk()
    root.withdraw()  # Hide the root window
    root.call('wm', 'attributes', '.', '-topmost', True)  # Bring dialog to front
    file_path = askopenfilename(
        title="Select an Audio File",
        filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.flac *.aac *.ogg")]
    )
    root.destroy()
    return file_path


def get_whisper_model():
    """
    Get the best available Whisper model based on system capabilities.
    """
    if torch.cuda.is_available():
        print("CUDA available - using large-v3 model for best accuracy")
        return "large-v3"
    else:
        print("CUDA not available - using medium model")
        return "medium"


def transcribe_with_whisper(file_path):
    """
    Transcribe audio using Whisper with optimal settings.
    """
    print("Starting transcription with Whisper...")
    
    model_name = get_whisper_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = whisper.load_model(model_name, device=device)
        
        # Use optimal transcription parameters
        result = model.transcribe(
            file_path,
            language=None,  # Auto-detect language
            task="transcribe",
            fp16=torch.cuda.is_available(),  # Use fp16 for CUDA
            verbose=False
        )
        
        print(f"Transcription completed. Number of segments: {len(result['segments'])}")
        return result
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        sys.exit(1)


def perform_speaker_diarization(file_path, hf_token, num_speakers):
    """
    Perform speaker diarization using the latest pyannote model.
    """
    print("Starting speaker diarization with Pyannote...")
    
    try:
        # Use the latest pyannote model (3.1+)
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
        
        # Perform diarization with improved parameters
        diarization = pipeline(
            file_path,
            num_speakers=num_speakers,
            min_speakers=1,
            max_speakers=num_speakers + 2  # Allow some flexibility
        )
        
        print(f"Diarization completed. Number of speaker segments: {len(diarization)}")
        return diarization
        
    except Exception as e:
        print("Failed to load pyannote model. Trying fallback...")
        try:
            # Fallback to older model
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization@2.1",
                use_auth_token=hf_token
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pipeline.to(device)
            diarization = pipeline(file_path, num_speakers=num_speakers)
            print(f"Diarization completed with fallback model. Number of speaker segments: {len(diarization)}")
            return diarization
        except Exception as e2:
            print("Both pyannote models failed to load.")
            print("Ensure that you have accepted the model's terms on Hugging Face and that your token is correct.")
            print(f"Error details: {e2}")
            sys.exit(1)


def improved_speaker_assignment(whisper_segments, diarization):
    """
    Improved speaker assignment algorithm that handles overlapping speech better.
    """
    print("Assigning speakers to transcription segments...")
    
    segments = []
    
    for segment in whisper_segments:
        start = segment['start']
        end = segment['end']
        text = segment['text'].strip()
        
        # Find all overlapping speaker segments
        overlapping_speakers = []
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Calculate overlap
            overlap_start = max(start, turn.start)
            overlap_end = min(end, turn.end)
            
            if overlap_start < overlap_end:  # There is overlap
                overlap_duration = overlap_end - overlap_start
                segment_duration = end - start
                overlap_ratio = overlap_duration / segment_duration
                
                overlapping_speakers.append((speaker, overlap_duration, overlap_ratio))
        
        if overlapping_speakers:
            # Sort by overlap ratio first, then by duration
            overlapping_speakers.sort(key=lambda x: (x[2], x[1]), reverse=True)
            speaker = overlapping_speakers[0][0]
        else:
            # Find the closest speaker segment
            closest_speaker = None
            min_distance = float('inf')
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Calculate distance to segment center
                segment_center = (start + end) / 2
                turn_center = (turn.start + turn.end) / 2
                distance = abs(segment_center - turn_center)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_speaker = speaker
            
            speaker = closest_speaker if closest_speaker else "Unknown"
        
        segments.append({
            'speaker': speaker,
            'start': start,
            'end': end,
            'text': text
        })
    
    return segments


def save_results(segments, file_path):
    """
    Save transcription results as readable text with speaker labels.
    """
    base_name = os.path.splitext(file_path)[0]
    
    # Save as readable text
    txt_file = base_name + "_transcription.txt"
    try:
        with open(txt_file, "w", encoding='utf-8') as f:
            current_speaker = None
            for segment in segments:
                if segment['speaker'] != current_speaker:
                    current_speaker = segment['speaker']
                    f.write(f"\n\n[{current_speaker}]:\n")
                f.write(f"{segment['text']} ")
        print(f"Transcription saved: {txt_file}")
    except Exception as e:
        print(f"Error saving transcription: {e}")
    
    return txt_file


def main():
    print("=== Modern Audio Transcription and Speaker Diarization ===\n")
    
    # Step 1: Get Hugging Face token securely
    hf_token = get_hf_token()
    print("Using provided Hugging Face token.\n")
    # Skip verification for now - will be checked when loading pyannote model
    
    # Step 2: Select Audio File
    file_path = select_file()
    if not file_path:
        print("No file selected. Exiting...")
        sys.exit(0)
    print(f"Selected file: {file_path}\n")
    
    # Step 3: Prompt for Number of Speakers
    num_speakers = get_number_of_speakers()
    print(f"Number of speakers set to: {num_speakers}\n")
    
    # Step 4: Sanitize Filename
    file_path = sanitize_filename(file_path)
    
    # Step 5: Convert to WAV if necessary
    file_path = convert_to_wav(file_path)
    
    # Step 6: Preprocess audio for better speaker recognition
    preprocessed_file = preprocess_audio(file_path)
    
    # Step 7: Transcribe with Whisper
    whisper_result = transcribe_with_whisper(preprocessed_file)
    
    # Step 8: Perform Speaker Diarization
    diarization = perform_speaker_diarization(preprocessed_file, hf_token, num_speakers)
    
    # Step 9: Combine results with improved algorithm
    segments = improved_speaker_assignment(whisper_result['segments'], diarization)
    
    print(f"Combined {len(segments)} transcription segments with diarization.\n")
    
    # Step 10: Show sample results
    print("Sample combined segments:")
    for i, seg in enumerate(segments[:5]):
        print(f"{i+1}. {seg}")
    print("\n")
    
    # Step 11: Save results
    txt_file = save_results(segments, file_path)
    
    # Cleanup preprocessed file
    try:
        os.remove(preprocessed_file)
        print("Cleaned up temporary preprocessed file.")
    except:
        pass
    
    print("\n=== Process Completed Successfully! ===")
    print(f"Output file: {txt_file}")


if __name__ == "__main__":
    main()
