import numpy as np
import DataParser
import VAD
import random
import librosa
from scipy import signal
import sys


class DataProvider:

    def __init__(self):
        for isTrain in [True, False]:
            self.isTrain = isTrain
            self.dataParser = DataParser.DataParser()
            self.vad = VAD.VAD()
            self.dataPathDictionary = self.__createDicitonaryForDataPath__()
            self.__makeEverySpectogramSame__()



    def hasNext(self, isTrain):
        if isTrain:
            return True if len(self.trainSpectograms) else False
        else:
            return True if len(self.testSpectograms) else False

    def next(self, isTrain):
        if isTrain:
            next_spectogram = self.trainSpectograms[0]
            del self.trainSpectograms[0]
            return next_spectogram
        else:
            next_spectogram = self.testSpectograms[0]
            del self.testSpectograms[0]
            return next_spectogram
    
    def returnSizeOfEverySpectogram(self):
        return self.sizeOfEverySpectogram

    def __makeEverySpectogramSame__(self):
        # setup toolbar
        sys.stdout.write("[")
        sys.stdout.flush()

        if self.isTrain:
            self.maxLength = 0
        dataSpectograms = []
        while self.__hasNext__():
            spectrogram, y = self.__next__()
            if self.maxLength < len(spectrogram):
                if self.isTrain:
                    self.maxLength = len(spectrogram)
                    print("new value of ", self.maxLength)
                else:
                    spectrogram = spectrogram[:self.maxLength]
                    print("resized to ", self.maxLength)
            dataSpectograms.append((spectrogram, y))
            sys.stdout.write("-")
            sys.stdout.flush()
        sys.stdout.write("]\n")

        sameShapeSpectograms = []
        sys.stdout.write("[")
        sys.stdout.flush()
        for spectrogram,y in dataSpectograms:
            if len(spectrogram) < self.maxLength:
                #print(np.matrix(spectrogram).shape)
                zeros = np.zeros((self.maxLength - len(spectrogram), len(spectrogram[0])))
                #print(zeros.shape)
                spectrogram = np.append(spectrogram,zeros, axis = 0)
                #print(np.matrix(spectrogram).shape)

            sameShapeSpectograms.append((np.ravel(spectrogram), y))
            sys.stdout.write("#")
            sys.stdout.flush()

        sys.stdout.write("]\n")
        self.sizeOfEverySpectogram = len(sameShapeSpectograms[0][0])
        if self.isTrain:
            self.trainSpectograms = sameShapeSpectograms
        else:
            self.testSpectograms = sameShapeSpectograms


    def __hasNext__(self):
        return True if len(self.dataPathDictionary.keys()) else False

    def __next__(self):
        dataPathDictionaryKeyLength = len(self.dataPathDictionary.keys())
        randomNumber = list(self.dataPathDictionary.keys())[random.randint(0,dataPathDictionaryKeyLength-1)]
        # randomNumber = list(self.dataPathDictionary.keys())[0]
        randomDataPathIndexInList = random.randint(0, len(self.dataPathDictionary[randomNumber])-1)
        # randomDataPathIndexInList = 0
        randomDataPathIndexInDataParser = self.dataPathDictionary[randomNumber][randomDataPathIndexInList]
        del self.dataPathDictionary[randomNumber][randomDataPathIndexInList]

        if len(self.dataPathDictionary[randomNumber]) == 0:
            del self.dataPathDictionary[randomNumber]
        
        randomDataPath = self.dataParser.return_data_path_on_coordinates(self.isTrain, randomNumber, randomDataPathIndexInDataParser)
        #print(randomDataPath)

        result = np.zeros(5)
        if self.isTrain:
            result[randomNumber - 1] = 1
        else:
            randomNumber = self.dataParser.parse_file_name_to_number(randomDataPath)
            #print(randomNumber)
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
        data_numpy_no_silence = self.vad.remove_silences(data_numpy, 0.02)
        freqs, times, spectrogram = self.__log_specgram__(data_numpy_no_silence, sampling_rate)
        mean = np.mean(spectrogram, axis=0)
        std = np.std(spectrogram, axis=0)
        if np.count_nonzero(std) != len(std):
            print(dataPath)
            print(spectrogram)
        spectrogram = (spectrogram - mean) / std
        if np.isnan(spectrogram[0][0]):
            print(dataPath)
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
