import numpy as np

SIDES = ["Left", "Right"]
AGGRESSIVE_DIFFERENCES_MULTIPLAYER = 0.5

class VAD:
    
    def remove_silences(self, numpy_data, aggressive = 1):
        max_amplitude = np.max(np.abs(numpy_data))
        current_frequency = numpy_data[0]
        freq_difference_allowed = AGGRESSIVE_DIFFERENCES_MULTIPLAYER * max_amplitude * aggressive
        numpy_data = self.remove_silence_on_side(numpy_data, SIDES[0], freq_difference_allowed)
        numpy_data = self.remove_silence_on_side(numpy_data, SIDES[1], freq_difference_allowed)
        return numpy_data


    def remove_silence_on_side(self, numpy_data, side, freq_difference_allowed):
        if side == SIDES[0]:
            iter_step = 1
            starting_point = 0 
            ending_point = len(numpy_data) - 1       
        else:
            iter_step = -1
            starting_point = len(numpy_data) - 1
            ending_point = 0

        silence_ending_point = -1
        last_freq = numpy_data[starting_point]
        for i in range(starting_point, ending_point, iter_step):
            if numpy_data[i] + freq_difference_allowed > last_freq and numpy_data[i] - freq_difference_allowed < last_freq:
                last_freq = numpy_data[i]
            else:
                silence_ending_point = i
                break
        
        if silence_ending_point < 0:
            print("NOT VALID DATA")
            assert "NOT VALID DATA"
        if side == SIDES[0]:
            if numpy_data[silence_ending_point:] is None:
                return self.remove_silence_on_side(numpy_data, side, freq_difference_allowed - 0.001)
            return numpy_data[silence_ending_point:]
        else:
            if numpy_data[:silence_ending_point] is None:
                return self.remove_silence_on_side(numpy_data, side, freq_difference_allowed - 0.001)
            return numpy_data[:silence_ending_point]
        

        