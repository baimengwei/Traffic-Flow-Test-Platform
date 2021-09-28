import os
import shutil

def change_roadnet(root_dir):
    list_file_dirs = os.listdir(root_dir)
    list_dirs = []

    source_file = None
    for file in list_file_dirs:
        if ".json" in file:
            source_file = os.path.join(root_dir, file)
        else:
            list_dirs += [file]
    if source_file is None:
        raise FileNotFoundError("check source_file")

    for file_dir in list_dirs:
        file_dir = os.path.join(root_dir, file_dir)
        list_files_ = os.listdir(file_dir)
        for file_str in list_files_:
            file_dir_ = os.path.join(file_dir, file_str)
            if "roadnet" in file_str:
                os.remove(file_dir_)
        shutil.copy(source_file, file_dir)



if __name__ == "__main__":
    os.chdir('../')
    root_dir = './data/anno_scenario'
    change_roadnet(root_dir)


    print("finished.")
