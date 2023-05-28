import sys
import os
import numpy as np
from scipy.io import wavfile
from PIL import Image

def unsigned_to_signed(u_data):
    s_data = (u_data - 32768).astype('int16')
    return s_data

def main(image_source):
    # Open the image
    img = Image.open(image_source)

    # Convert the image data into an array
    audio_data_fit = np.array(img)

    # Flatten the array
    audio_data_flat = audio_data_fit.flatten()

    # Deinterleave the audio data
    audio_data_L_flat = audio_data_flat[0::2]
    audio_data_R_flat = audio_data_flat[1::2]

    # Convert audio data
    audio_data_L = unsigned_to_signed(audio_data_L_flat)
    audio_data_R = unsigned_to_signed(audio_data_R_flat)

    # Stack audio data into 2D array
    audio_data = np.column_stack((audio_data_L, audio_data_R))

    # Save as .wav file
    output_filename = os.path.splitext(image_source)[0] + "_Decoded.wav"
    wavfile.write(output_filename, 44100, audio_data)

    print(f"Output Audio: {output_filename}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <encoded_image.png>")
        sys.exit()

    image_file = sys.argv[1]
    main(image_file)
