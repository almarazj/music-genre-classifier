import os
import numpy as np
import librosa
from tqdm import tqdm

def save_to_npz(S: np.ndarray, 
                npz_genre_path: str,
                file_name: str, 
                iter: int) -> None:
    """
    Saves a spectrogram to a .npz file.

    Parameters
    ----------
    S : np.ndarray
        The spectrogram data to be saved.
    npz_genre_path : str
        The path where the .npz file will be saved.
    file_name : str
        Name of the audio file.
    iter : int
        Number of segment.
    """
    npz_path = os.path.join(npz_genre_path, f"{os.path.splitext(file_name)[0]}-{iter}.npz")
    np.savez_compressed(npz_path, spectrogram=S)

def split_audio(y: np.ndarray,
                sr: int,
                segment_length: int = 3) -> list:    
    """
    Splits an audio file into a specified number of equal segments.

    Parameters
    ----------
    y : np.ndarray
        Audio signal to split.
    sr : int
        Sample rate of the audio signal.
    n_segments : int
        The number of segments to split the audio into (default is 10).
    
    Returns
    -------
    list of np.ndarray
        A list containing the audio segments as numpy arrays.
    """
    samples_per_segment = sr * segment_length
    n_segments = len(y) // samples_per_segment
    remainder = len(y) % samples_per_segment
    segments = []

    start = 0 
    for i in range(n_segments):
        end = start + samples_per_segment

        if i < remainder:
            end += 1
        segment = y[start:end]
        segments.append(segment)
        start = end

    return segments

def load_audio(file_path: str) -> tuple[np.ndarray, int] | tuple[None, None]:
    """
    Loads an audio file and returns the audio time series and the sample rate.

    Parameters
    ----------
    file_path : str
        The path to the audio file to be loaded.

    Returns
    -------
    tuple[np.ndarray, int]
        A tuple containing:
        - y: np.ndarray
            The audio time series as a NumPy array.
        - sr: int
            The sample rate of the audio file.

    Raises
    ------
    Exception
        If there is an error loading the file, the exception is caught and an error message is printed.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def get_mel_spectrogram(y: np.ndarray,
                        sr: int,
                        n_fft: int=1024,
                        hop_length: int=512,
                        n_mels: int=128) -> np.ndarray:
    
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, 
                                       hop_length=hop_length, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
        
    assert S_dB.shape == (n_mels, 130), \
        f"Inconsistent spectrogram shape: {S_dB.shape}" 
    
    return S_dB             
    
def preprocess_file(file_path: str,
                    file_name: str,
                    npz_genre_path: str) -> None:
    """
    Takes a wav file path as input, divides the signal in n segments and returns the mel spectrogram
    of every segment in npz format.

    Parameters
    ----------
    file_path : str
        File path of the wav file.
    file_name : str
        Name of te wav file.
    npz_genre_path: str
        Path to save the npz file.
    segment_length : int
        Length of the segment to split the audio into.
    """
    y, sr = load_audio(file_path)
    if y is None or sr is None:
        print(f"Skipping file {file_path} due to loading error.")
        return
    
    segments = split_audio(y, sr)
    
    for i, segment in enumerate(segments):
        S_dB = get_mel_spectrogram(segment, sr)  
        save_to_npz(S_dB, file_name=file_name, npz_genre_path=npz_genre_path, iter=i)

def preprocessing(dataset_path: str,
                  npz_path : str) -> None:
    """
    Preprocessing of the dataset to be used for model training.
    
    Parameters
    ----------
    dataset_path : str
        Path containing the dataset organized in folders according to music genre.
    npz_path : str
        Path to save the .npz files containing spectrogram information of the audio files.
    """
    for genre in os.listdir(dataset_path):
        
        genre_path = os.path.join(dataset_path, genre)
        npz_genre_path = os.path.join(npz_path, genre)
        
        total_files = sum([len(files) for r, d, files in os.walk(genre_path) if any(f.endswith('.wav') for f in files)])
        with tqdm(total=total_files, desc="Processing audio files corresponding to the " + genre + " musical genre") as pbar:
            if not os.path.exists(npz_genre_path):
                os.makedirs(npz_genre_path)
            if os.path.isdir(genre_path):
                for file_name in os.listdir(genre_path):
                    if file_name.endswith('.wav'):
                        file_path = os.path.join(genre_path, file_name)
                        preprocess_file(file_path, file_name, npz_genre_path)
                        
                        pbar.update(1)
                        
if __name__ == '__main__':
    dataset_path = 'dataset\\genres_original'
    npz_path = 'dataset\\genres_mel_npz'
    preprocessing(dataset_path, npz_path)