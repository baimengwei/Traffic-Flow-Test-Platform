import os
import pickle
if __name__=='__main__':
    print('start')
    with open('conf_traffic_intersection_1_1.pkl', mode='rb') as f:
        cf = pickle.load(f)

    list_folder = os.listdir(('.'))
    list_file_name = []
    for folder in list_folder:
        if '.py' in folder:
            list_file_name.append(folder)

