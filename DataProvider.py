import numpy as np
import DataParser
import VAD
import random
import librosa
from scipy import signal

class DataProvider:

    def __init__(self, isTrain = True):
        self.isTrain = isTrain
        self.dataParser = DataParser.DataParser()
        self.vad = VAD.VAD()
        self.dataPathDictionary = self.__createDicitonaryForDataPath__(isTrain)
        while self.hasNext():
            print(self.next().shape)


    def hasNext(self):
        return True if len(self.dataPathDictionary.keys()) else False

    def next(self):
        dataPathDictionaryKeyLength = len(self.dataPathDictionary.keys())
        randomNumber = list(self.dataPathDictionary.keys())[random.randint(0,dataPathDictionaryKeyLength-1)]
        randomDataPathIndexInList = random.randint(0, len(self.dataPathDictionary[randomNumber])-1)
        randomDataPathIndexInDataParser = self.dataPathDictionary[randomNumber][randomDataPathIndexInList]
        del self.dataPathDictionary[randomNumber][randomDataPathIndexInList]

        if len(self.dataPathDictionary[randomNumber]) == 0:
            del self.dataPathDictionary[randomNumber]
        
        randomDataPath = self.dataParser.return_data_path_on_coordinates(self.isTrain, randomNumber, randomDataPathIndexInDataParser)
        print(randomDataPath)
        return self.__getDataFromPath__(randomDataPath)


    
    def __createDicitonaryForDataPath__(self, isTrain):
        dataPathDictionary = dict()
        for number in range(1,6,1):
            dataCountOfNumber = self.dataParser.length_of_files_in_folder(isTrain,number)
            dataPathDictionary[number] = list(range(dataCountOfNumber-1))
        return dataPathDictionary




    def __getDataFromPath__(self, dataPath):
        data_numpy, sampling_rate = librosa.load(dataPath, sr=16000)
        data_numpy_no_silence = self.vad.remove_silences(data_numpy, 0.01)
        freqs, times, spectrogram = self.__log_specgram__(data_numpy_no_silence, sampling_rate)
        mean = np.mean(spectrogram, axis=0)
        std = np.std(spectrogram, axis=0)
        spectrogram = (spectrogram - mean) / std
        return spectrogram


    def __log_specgram__(self, audio, sample_rate, window_size=20, step_size=10, eps=1e-10):
        nperseg = int(round(window_size * sample_rate / 1e3))
        noverlap = int(round(step_size * sample_rate / 1e3))
        freqs, times, spec = signal.spectrogram(audio, fs=sample_rate, window='hann', nperseg=nperseg, noverlap=noverlap, detrend=False)
        return freqs, times, np.log(spec.T.astype(np.float32) + eps)



if __name__ == "__main__":
    DataProvider()
