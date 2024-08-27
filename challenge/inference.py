import os
import numpy as np
import torch
import torchaudio
import librosa
from preprocessing import split_audio
from models.cnn import CNN
import data_managment.dataset as dataset
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def convert_mp3_to_wav(mp3_folder: str, wav_folder: str, target_sample_rate: int) -> None:
    """
    Convert all MP3 files in a folder to WAV format with a specific sample rate using torchaudio.

    Parameters
    ----------
    mp3_folder : str
        The folder containing MP3 files to convert.
    wav_folder : str
        The folder where the converted WAV files will be saved.
    target_sample_rate : int
        The target sample rate for the output WAV files.
    """

    if not os.path.exists(wav_folder):
        os.makedirs(wav_folder)
    
    for file_name in os.listdir(mp3_folder):
        if file_name.endswith('.mp3'):
            mp3_path = os.path.join(mp3_folder, file_name)
            wav_path = os.path.join(wav_folder, os.path.splitext(file_name)[0] + '.wav')

            if not os.path.exists(wav_path):
                # Load MP3 file with torchaudio
                waveform, original_sample_rate = torchaudio.load(mp3_path, format="mp3")
                
                # Resample to the target sample rate if necessary
                if original_sample_rate != target_sample_rate:
                    resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
                    waveform = resampler(waveform)
                
                # Save as WAV file with the target sample rate
                torchaudio.save(wav_path, waveform, target_sample_rate)
                print(f"Converted {file_name} to WAV format with sample rate {target_sample_rate} Hz.")

def preprocess_file(file_path: str,
                    segment_length: int = 3,
                    n_mels: int = 128,
                    n_fft: int = 1024,
                    hop_length: int = 512):
    """
    Takes a wav file path as input, divides the signal in n segments and returns the mel spectrogram
    of every segment in npz format.

    Parameters
    ----------
    file_path : str
        File path of the wav file.
    segment_length : int
        Length of the segment to split the audio into.
    n_mels : int
        Number of Mel bands to generate.
    n_fft : int
        Length size of the FFT window.
    hop_length : int
        Number of samples between succesive frames.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return
    
    segments = split_audio(y, sr, segment_length)
    expected_segment_length = sr * segment_length
    spectrograms = []

    for i, segment in enumerate(segments):
        segment = librosa.util.fix_length(segment, size=expected_segment_length)
        
        S = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=n_fft, 
                                            hop_length=hop_length, n_mels=n_mels)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        assert S_dB.shape == (n_mels, 130), \
            f"Inconsistent spectrogram shape: {S_dB.shape}"       
        
        spectrograms.append(S_dB)

    spectrograms_tensors = []
    for spectrogram in spectrograms:       
        tensor = torch.from_numpy(spectrogram).unsqueeze(0) # Pytorch tensor (1, H, W)
        tensor = tensor.unsqueeze(0) # Pytorch tensor (1, 1, H, W)
        spectrograms_tensors.append(tensor)
        
    return spectrograms_tensors

def predict_genre(song_path : str,
                  model,
                  class_mapping: dict) -> dict:
    """
    Predicts genre of the input song.

    Parameters
    ----------
    song_path : str
        Path of the song to predict the genre.
    model : torch.nn.Module
        Model used to predict genre.
    class_mapping : dict
        Dict of genres and their indexes.
    
    Returns
    -------
    Prediction : str
        Predicted genre of the song.
    """

    spectrogram_tensors = preprocess_file(song_path)

    predictions = []
    model.eval()

    for tensor in spectrogram_tensors:
        with torch.no_grad():
            prediction = model(tensor)
            predictions.append(prediction.numpy())

    prediction = np.mean(predictions, axis=0) # (1, 10)
    prediction = prediction.squeeze(0) # (10, )
    predicted_index = np.argmax(prediction)
    genre_predicted = class_mapping[predicted_index]
    return genre_predicted

if __name__ == '__main__':

    # Load the model
    cnn = CNN(input_shape=(1, 128, 130))
    state_dict = torch.load('results/music_genre_classifier.pth')
    cnn.load_state_dict(state_dict)                        
    
    class_mapping = dataset.genres_map
    wav_folder='songs\\wav'
    print("Test with 9 songs of different genres:")
    print('--------------------------------------')
    convert_mp3_to_wav(mp3_folder='songs\\mp3', wav_folder='songs\\wav', target_sample_rate=22050)
    for file_name in os.listdir(wav_folder):
        file_path = os.path.join(wav_folder, file_name)
        genre_predicted = predict_genre(file_path, cnn, class_mapping)
        print(f'Genre of the song {file_name}: {genre_predicted}')
    