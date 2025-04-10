# FLAC to AIFF converter
# Desciption:   Converts all .flac files into .aiff or .wav
#
# Author:       Jérôme Roy
# Date:         22.03.23
# Update:       22.03.23

import os
import soundfile as sf
import librosa



directories = [
    "/home/var/Desktop/Mohor/InferAnomaly/SlurredSpeech-Dataset-Test",
    "/home/var/Desktop/Mohor/InferAnomaly/SlurredSpeech-Dataset-Test(Hindi)",
    "/home/var/Desktop/Mohor/InferAnomaly/SlurredSpeech-Dataset-Train",
    "/home/var/Desktop/Mohor/InferAnomaly/SlurredSpeech-Dataset-Train(Hindi)"
]
for top_dir in directories:


    print(f"\nProcessing top-level directory: {top_dir}")

    for root, _, files in os.walk(top_dir):
        for file_name in files:
            if file_name.endswith(".flac"):
        # Build the full path to the FLAC file
                flac_path = os.path.join(root, file_name)

                # Build the full path to the AIFF file
                wav_path = os.path.join(root, os.path.splitext(file_name)[0] + ".wav") #change to .wav if you want .wav files

                # Load the FLAC file and write it to a AIFF file
                print(flac_path)
                data, samplerate = librosa.load(flac_path)
                
                print("lll")
                sf.write(wav_path, data, samplerate)
