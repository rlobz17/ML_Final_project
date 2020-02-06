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
        self.dataPathDictionary = self.__createDicitonaryForDataPath__()
        # while self.hasNext():
        #     spectrogram, number_list = self.next()
        #     print(number_list, ') ', spectrogram.shape)
        #print(self.dataParser.__return_data_pathes__())


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
        
        result = np.zeros(5)
        if self.isTrain:
            result[randomNumber - 1] = 1
        else:
            randomNumber = self.dataParser.parse_file_name_to_number(randomDataPath)
            result = randomNumber
        return self.__getDataFromPath__(randomDataPath), result

    
    def __createDicitonaryForDataPath__(self):
        dataPathDictionary = dict()
        if self.isTrain:
            for number in range(1,6,1):
                dataCountOfNumber = self.dataParser.length_of_files_in_folder(self.isTrain, number)
                dataPathDictionary[number] = list(range(dataCountOfNumber))
            return dataPathDictionary
        else:
            dataCountOfNumber = self.dataParser.length_of_files_in_folder(self.isTrain, 0)
            dataPathDictionary[0] = list(range(dataCountOfNumber))
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
    dataProvider = DataProvider(False)
    while dataProvider.hasNext():
        spectrogram, result = dataProvider.next()
        print(spectrogram.shape, result)
