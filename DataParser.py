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

    def return_one_random_path_file(self, isTrain: bool, number_voice: int):
        random_file_index = random.randint(0, self.length_of_files_in_folder(isTrain,number_voice))
        return self.data_pathes[DATA_SET_GROUP[0]][number_voice][random_file_index] if isTrain else self.data_pathes[DATA_SET_GROUP[1]][number_voice][random_file_index]

    def return_data_pathes(self):
        return self.data_pathes

    def length_of_files_in_folder(self, isTrain: bool, number_voice: int):
        return len(self.data_pathes[DATA_SET_GROUP[0]][number_voice]) if isTrain else len(self.data_pathes[DATA_SET_GROUP[1]][number_voice])

    def return_data_path_on_coordinates(self, isTrain: bool, number_voice: int, index_in_data: int):
        return self.data_pathes[DATA_SET_GROUP[0]][number_voice][index_in_data] if isTrain else self.data_pathes[DATA_SET_GROUP[1]][number_voice][index_in_data]

    def parse_file_name_to_creator_name(self, file_path: str):
        splitted_path = file_path.split("/")
        file_name = splitted_path[len(splitted_path)-1]
        first_name, last_name, _ = file_name.split("-")
        return first_name,last_name
        

if __name__ == "__main__":
    data_parser = DataParser()
    #print(data_parser.return_one_random_path_file(True, 1))
    #print(data_parser.return_data_pathes())
    file_path = data_parser.return_data_path_on_coordinates(True, 1, 2)
    print("parsing file on path:",file_path)
    first_name, last_name = data_parser.parse_file_name_to_creator_name(file_path)
    print(first_name, last_name, "is this file creator")