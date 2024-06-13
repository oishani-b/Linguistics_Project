import os
import csv
import subprocess
from pydub import AudioSegment
from google.cloud import speech
import whisper
import requests
import pandas as pd

def is_mono(wav_file_path):
    os.environ["PATH"] += os.pathsep + '/opt/homebrew/bin'
    """Check if a .wav file is mono."""
    audio = AudioSegment.from_wav(wav_file_path)
    return audio.channels == 1



def convert_ogg_to_wav_ffmpeg(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.ogg'):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename[:-4] + '.wav')

            # Command to convert the file using ffmpeg
            command = ['ffmpeg', '-i', input_file_path, output_file_path]
            
            # Execute the command
            try:
                subprocess.run(command, check=True, stderr=subprocess.PIPE)
                print(f"Converted {filename} to WAV format.")
            except subprocess.CalledProcessError as e:
                print(f"Failed to convert {filename}: {e.stderr.decode()}")

# Example usage
'''
input_folder = "/Users/oishanibandopadhyay/Documents/Research Project ERC 92 Honors/audio"  # Update this path
output_folder = "/Users/oishanibandopadhyay/Documents/Research Project ERC 92 Honors/audio"  # Update this path
convert_ogg_to_wav_ffmpeg(input_folder, output_folder)
'''


# Checks if the audio file is mono and converts to mono if it is not
def convert_to_mono_if_needed(audio_file_path):
    os.environ["PATH"] += os.pathsep + '/opt/homebrew/bin'

    # Load the audio file
    audio = AudioSegment.from_file(audio_file_path)
    
    # Check if the audio is already mono
    if audio.channels == 1:
        print(f"The file {audio_file_path} is already mono.")
    else:
        # Convert to mono by averaging the channels
        mono_audio = audio.set_channels(1)
        
        # Save the mono audio to the same file or a new file
        mono_audio.export(audio_file_path)
        print(f"Converted {audio_file_path} to mono.")

'''
# Google transcription function
def transcribe_audio_google(speech_file):
    convert_to_mono_if_needed(speech_file)
    os.environ["PATH"] += os.pathsep + '/opt/homebrew/bin'
    """Transcribe the given audio file using Google Cloud Speech-to-Text."""
    client = speech.SpeechClient()

    with open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=48000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)
    result = " ".join(result.alternatives[0].transcript for result in response.results)

    # Generate the name of the output file by replacing the audio file extension with .txt
    output_file_path = os.path.splitext(speech_file)[0] + "_google" + ".txt"

    # Save the transcription to a text file
    with open(output_file_path, 'w') as file:
        file.write(result)
    print(f"Google transcription saved to {output_file_path}")

    return result
'''

from google.cloud import speech_v1p1beta1 as speech
import os

def transcribe_audio_google_long(audio_file_path):
    # Create a speech client
    client = speech.SpeechClient()

    # Full path of the audio file, assumes file is in the same folder as the script
    with open(audio_file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True
    )

    # Using the long-running recognize method for longer audio files
    operation = client.long_running_recognize(config=config, audio=audio)
    print("Waiting for operation to complete...")
    response = operation.result(timeout=90)

    # Print and return all the results
    transcripts = []
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        transcripts.append(result.alternatives[0].transcript)

    # Join all parts of the transcripts into one string
    full_transcript = ' '.join(transcripts)
    return full_transcript

'''
# Example usage
audio_file_path = '/path/to/your/long/audiofile.wav'
try:
    transcript = transcribe_audio_google_long(audio_file_path)
    print("Transcription:", transcript)
except Exception as e:
    print("Failed to transcribe:", e)


google_transcript = transcribe_audio_google_long("laptop_earphone_test_recording_1.wav")
print(google_transcript)

#print(transcribe_audio_google("laptop_earphone_test_recording_1.wav"))
'''

# Whisper transcription function
def transcribe_audio_whisper(file_path):
    # Load the Whisper model
    model = whisper.load_model("base")

    # Load and transcribe the audio file
    result = model.transcribe(file_path)

    # Extract and print the transcription
    transcription = result['text']
    print("Transcription:", transcription)

    # Generate the name of the output file by replacing the audio file extension with .txt
    output_file_path = os.path.splitext(file_path)[0] + "_whisper" + ".txt"

    # Save the transcription to a text file
    with open(output_file_path, 'w') as file:
        file.write(transcription)
    print(f"Whisper transcription saved to {output_file_path}")

    return transcription

#print(transcribe_audio_whisper("laptop_earphone_test_recording_1.wav"))


#Microsoft Azure transcription function
import requests
import os

def transcribe_audio_azure(audio_file_path, subscription_key, service_region):
    # Construct the endpoint URL
    endpoint = f"https://{service_region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
    
    # Headers and params for the request
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Content-Type': 'audio/wav; codecs=audio/pcm; samplerate=16000',
        'Accept': 'application/json'
    }
    params = {
        'language': 'en-US',
        'format': 'detailed'
    }
    
    # Read the audio file
    with open(audio_file_path, 'rb') as audio_file:
        audio_data = audio_file.read()
    
    # Make the request to the API
    response = requests.post(endpoint, headers=headers, params=params, data=audio_data)
    response.raise_for_status()  # Raises an exception for HTTP errors
    
    # Parse the response
    result = response.json()
    try:
        # Extract the best transcription result
        transcription = result['NBest'][0]['Lexical']  # Adjust the key based on the actual response structure
    except (KeyError, IndexError):
        print("Failed to parse transcription results.")
        print(result)  # Log the full response to help diagnose the issue
        return

    print("Transcription:", transcription)
    
    '''
    # Save the transcription to a text file
    output_file_path = os.path.splitext(audio_file_path)[0] + "_microsoft.txt"
    with open(output_file_path, 'w') as file:
        file.write(transcription)
    print(f"Transcription saved to {output_file_path}")
    '''

    return transcription

'''
if __name__ == "__main__":
    audio_file_path = '/Users/oishanibandopadhyay/Documents/Research Project ERC 92 Honors/laptop_earphone_test_recording_1.wav'  # Update this path
    subscription_key = '488c1ac4f68d439eb42df9549325b9d8'  # Replace with your subscription key
    service_region = 'westus'  # Replace with your service region, e.g., 'westus'
    transcribe_audio_azure(audio_file_path, subscription_key, service_region)
'''




def transcribe_folder(audio_folder_path):
    print("Starting transcription process...")
    # Prepare the CSV file to store the results
    output_csv_path = os.path.join(audio_folder_path, 'transcription_results.csv')
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['audio_filepath', 'whisper_transcript', 'azure_transcript'])

        file_count = 0  # To count the number of audio files processed

        convert_ogg_to_wav_ffmpeg(audio_folder_path, audio_folder_path)

        # Walk through the folder containing audio files
        for root, dirs, files in os.walk(audio_folder_path):
            for filename in files:
                if filename.lower().endswith(('.wav', '.mp3', '.m4a')):
                    file_count += 1
                    audio_file_path = os.path.join(root, filename)
                    
                    print(f"Processing file: {audio_file_path}")

                    # Simulated transcription functions, replace with actual calls
                    #google_transcript = transcribe_audio_google_long(audio_file_path)
                    whisper_transcript = transcribe_audio_whisper(audio_file_path)

                    subscription_key = '488c1ac4f68d439eb42df9549325b9d8'  # Replace with your subscription key
                    service_region = 'westus'  # Replace with your service region, e.g., 'westus'
                    azure_transcript = transcribe_audio_azure(audio_file_path, subscription_key, service_region)
                    
                    # Write the results to the CSV file
                    writer.writerow([audio_file_path, whisper_transcript, azure_transcript])

        if file_count == 0:
            print("No audio files found. Check the directory path and file extensions.")

    print(f"Transcription results saved to {output_csv_path}")

# Specify the correct path to your folder containing audio files
audio_folder_path = '/Users/oishanibandopadhyay/Documents/UCSD/LIGN 168/Final_Project_Files'
transcribe_folder(audio_folder_path)
