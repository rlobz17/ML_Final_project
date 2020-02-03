#####################################
###### STATIC FINAL VARIABLES #######
#####################################
PATH_TO_DATA_FOLDER = "./Data"
DATA_SET_GROUP = ["Train", "Test"]
#####################################

import glob
import random


class DataParser:

    def __init__(self):
        self.data_pathes = self.all_wav_files_path()

    def all_wav_files_path(self, path_to_data_folder = PATH_TO_DATA_FOLDER):
        data_pathes = dict()

        for data_set_group in DATA_SET_GROUP:
            files_path = PATH_TO_DATA_FOLDER + "/" + data_set_group
            set_group_data = dict()
            for i in range(5):
                files_path_with_format = files_path + "/" + str(i+1) + "/*.wav"
                files_in_the_path = glob.glob(files_path_with_format)
                set_group_data[i+1] = files_in_the_path
            data_pathes[data_set_group] = set_group_data

        return data_pathes

    def return_one_random_path_file(self, number_voice: int):
        random_file_index = random.randint(0, len(self.data_pathes["Train"][number_voice])-1)
        one_random_file_path = self.data_pathes["Train"][number_voice][random_file_index]
        return one_random_file_path

    def return_data_pathes(self):
        return self.data_pathes


if __name__ == "__main__":
    data_parser = DataParser()
    print(data_parser.return_one_random_path_file(1))
    print(data_parser.return_data_pathes())