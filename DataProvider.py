import numpy as np
import DataParser
import VAD
import random
import librosa
from scipy import signal
import sys


MULTIPLY_FOR_EVERY_BACKGROUND_VOICE = 3
N_MFCC = 20

class DataProvider:

    def __init__(self, isTrain, maxLength = 0, n_mfcc = N_MFCC):
        if isTrain in [True, False]:
            self.N_MFCC = n_mfcc
            self.isTrain = isTrain
            self.maxLength = maxLength
            self.dataParser = DataParser.DataParser()
            self.vad = VAD.VAD()
            self.dataPathDictionary = self.__createDicitonaryForDataPath__()
            self.__create_random_noise_files__()
            self.__makeEverySpectogramSame__()


    def hasNext(self):
        if self.isTrain:
            return True if len(self.trainSpectograms) else False
        else:
            return True if len(self.testSpectograms) else False

    def next(self):
        if self.isTrain:
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

        dataSpectograms = []
        while self.__hasNext__():
            spectrograms, y = self.__next__()
            for spectrogram in spectrograms:
                if self.maxLength < len(spectrogram):
                    if self.isTrain:
                        self.maxLength = len(spectrogram)
                        print("new value of ", self.maxLength)
                    else:
                        spectrogram = spectrogram[:self.maxLength]
                        print("resized to ", self.maxLength)
                #print(np.matrix(spectrogram).shape)
                dataSpectograms.append((spectrogram, y))
            sys.stdout.write("-")
            sys.stdout.flush()
        sys.stdout.write("]\n")

        sameShapeSpectograms = []
        sys.stdout.write("[")
        sys.stdout.flush()
        for spectrogram,y in dataSpectograms:
            #print(np.matrix(spectrogram).shape)
            if spectrogram.shape[0] < self.maxLength:
                #print(spectrogram.shape)
                zeros = np.zeros((self.maxLength - spectrogram.shape[0], spectrogram.shape[1]))
                #print(zeros.shape)
                spectrogram = np.append(spectrogram, zeros, axis = 0)
                #print(np.matrix(spectrogram).shape)
            
            if self.isTrain:
                resultVoices = self.__add_random_noise__(spectrogram)
                for spectrogram_voice in resultVoices:
                    sameShapeSpectograms.append((spectrogram_voice, y))
            else:
                sameShapeSpectograms.append((spectrogram, y))
            sys.stdout.write("#")
            sys.stdout.flush()
        sys.stdout.write("]\n") 

        self.sizeOfEverySpectogram = len(sameShapeSpectograms[0][0])
        if self.isTrain:
            self.trainSpectograms = sameShapeSpectograms
        else:
            self.testSpectograms = sameShapeSpectograms


    def __create_random_noise_files__(self):
        if self.isTrain:
            background_voices = self.dataParser.helper_files_path()
            self.background_voices = list()
            for background_voice_path in background_voices:
                print("reading helper data on path:", background_voice_path)
                data_numpy, sampling_rate = librosa.load(background_voice_path, sr=16000)
                self.background_voices.append(data_numpy)


    def __add_random_noise__(self, voice):
        result = list()
        result.append(voice)
        # for i in range(10):
        #     randoNoise = random.random()
        #     addition = np.full(voice.shape, randoNoise)
        #     newVoice = voice + addition
        #     result.append(newVoice)
        #     newVoice = voice - addition
        #     result.append(newVoice)
        return result

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

        
        if self.isTrain:
            result = np.zeros(5)
            result[randomNumber - 1] = 1
        else:
            randomNumber = self.dataParser.parse_file_name_to_number(randomDataPath)
            result = (randomDataPath, randomNumber)
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
        data_numpy = librosa.effects.percussive(data_numpy)
        #data_numpy_no_silence = self.vad.remove_silences(data_numpy, 0.3)
        data_numpy_no_silence,_ = librosa.effects.trim(data_numpy, top_db=18)
        datas = self.__add_background_voice__(data_numpy_no_silence)
        result = list()
        for data in datas:
            spectrogram = self.__log_specgram__(data, sampling_rate)
            mean = np.mean(spectrogram, axis=0)
            std = np.std(spectrogram, axis=0)
            zeros = np.where(std == 0)[0]
            for zero_index in zeros:
                std[zero_index] = spectrogram[0][zero_index]
            spectrogram = (spectrogram - mean) / std
            result.append(spectrogram)
        return result


    def __log_specgram__(self, audio, sample_rate):
        mfccs = librosa.feature.mfcc(audio, sr=sample_rate, n_mfcc = self.N_MFCC)
        return np.matrix(mfccs).T

    def __add_background_voice__(self, numpy_data):
        result = list()
        if self.isTrain:
            result.append(numpy_data)
            result.append(librosa.effects.time_stretch(numpy_data, 0.9))
            result.append(librosa.effects.time_stretch(numpy_data, 0.8))
            result.append(librosa.effects.time_stretch(numpy_data, 1.1))
            result.append(librosa.effects.time_stretch(numpy_data, 1.2))

            for i in range(len(self.background_voices)):
                for background_voice_counter in range(MULTIPLY_FOR_EVERY_BACKGROUND_VOICE):
                    background_voice_index = i
                    background_voice = self.background_voices[background_voice_index]
                    numpy_data_length = len(numpy_data)
                    start_index_in_background_voice = random.randint(0, len(background_voice)-1-numpy_data_length)
                    background_voice_data_to_add = background_voice[start_index_in_background_voice:start_index_in_background_voice+numpy_data_length]
                    result.append(numpy_data*0.95 + background_voice_data_to_add*0.05)
             
        else:
            result.append(numpy_data)
        return result



if __name__ == "__main__":
    dataProvider = DataProvider(False)
    while dataProvider.hasNext():
        spectrogram, result = dataProvider.next()
        print(spectrogram.shape, result)
