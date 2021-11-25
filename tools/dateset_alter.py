import os
import shutil


def to_same_env():
    os.chdir('../data/cityflow_scenario')
    list_file = os.listdir('.')
    for folder in list_file:
        if 'hangzhou' not in folder:
            continue
        detail_two_file = os.listdir(folder)
        [os.remove(os.path.join(folder, file_name))
         for file_name in detail_two_file
         if 'roadnet' in file_name]
        shutil.copy('../../tools/roadnet_p4a_lt.json', folder)
    print('finished!')


if __name__ == '__main__':
    to_same_env()
