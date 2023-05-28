import sys
import math
import numpy as np
import os
import warnings
from scipy.io import wavfile
from PIL import Image, PngImagePlugin
from tinytag import TinyTag

image_encode_format = "I;16"
image_compression_level = 9
separator_line_thickness = 0.25

def signed_to_unsigned(s_data):
    u_data = (s_data + 32768).astype('uint16')
    return u_data

def text_to_binary(string):
    return ''.join(format(ord(i), '08b') for i in string)

def binary_to_grayscale(binary_string):
    array = []
    for i in binary_string:
        if i == '0':
            array.append(0)
        else :
            array.append(65535)
    return array

def binary_to_string(bin_string):
    n = int(bin_string, 2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode()

def main(audio_source):
    warnings.filterwarnings('ignore')

    # Read audio file
    try:
        samplerate, data = wavfile.read(audio_source)
    except ValueError:
        print('File format unsupported')
        sys.exit()

    # Extract raw audio information
    bit_depth = data.dtype

    if bit_depth == 'int16':
        if data.ndim == 2:
            data_L = data[:, 0]
            data_R = data[:, 1]
        else:
            print("Audio file is not stereo")
            sys.exit()
    else:
        print("Bit Depth unsupported")
        sys.exit()

    # Prepare audio data
    audio_data_L = signed_to_unsigned(data_L)
    audio_data_R = signed_to_unsigned(data_R)

    # Calculate image dimensions
    number_of_samples = len(audio_data_L)
    dimension = math.sqrt(number_of_samples)  # Calculate dimension for a single channel
    dimension_fit = math.ceil(dimension)
    total_samples = dimension_fit**2
    extra_samples = total_samples - number_of_samples

    # Pad the audio data to fit image dimensions
    audio_data_L_fit = np.pad(audio_data_L, (0, extra_samples), 'constant')
    audio_data_R_fit = np.pad(audio_data_R, (0, extra_samples), 'constant')

    # Interleave the two channels
    audio_data_fit = np.empty((audio_data_L_fit.size + audio_data_R_fit.size,), dtype=audio_data_L_fit.dtype)
    audio_data_fit[0::2] = audio_data_L_fit
    audio_data_fit[1::2] = audio_data_R_fit

    # Calculate new image dimensions
    total_samples_interleaved = len(audio_data_fit)
    dimension_interleaved = math.sqrt(total_samples_interleaved)
    dimension_fit_interleaved = math.ceil(dimension_interleaved)

    # Pad the interleaved data to fit the image dimensions
    extra_samples_interleaved = dimension_fit_interleaved**2 - total_samples_interleaved
    audio_data_fit = np.pad(audio_data_fit, (0, extra_samples_interleaved), 'constant')

    # Create and save the image
    img = Image.fromarray(audio_data_fit.reshape(dimension_fit_interleaved, dimension_fit_interleaved), image_encode_format)
    output_filename = os.path.splitext(audio_source)[0] + "_Encoded.png"
    img.save(output_filename, "PNG", compression_level=image_compression_level)

    # Print compression ratio
    audio_size = os.path.getsize(audio_source)
    image_size = os.path.getsize(output_filename)
    compression_ratio = 100 - ((image_size / audio_size) * 100)
    print(f"Compression ratio: {compression_ratio}%")

    print(f"Output Image: {output_filename}")



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <audio_file.wav>")
        sys.exit()

    audio_file = sys.argv[1]
    main(audio_file)
