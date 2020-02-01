#####################################
###### STATIC FINAL VARIABLES #######
#####################################
PATH_TO_DATA_FOLDER = "./Data"
DATA_SET_GROUP = ["Train", "Test"]
#####################################

from os import walk
import glob


class DataParser:

    def __init__(self):
        self.data_pathes = self.all_wav_files_path()
        print self.data_pathes

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


if __name__ == "__main__":
    data_parser = DataParser()
