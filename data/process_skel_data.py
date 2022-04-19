import os
import pickle
import numpy as np


def find_all(action, path):
    result = []
    for file in os.listdir(path):
        if file[16:20] == action:
            result.append(file)
    return result


def extract_skel_data(data, body_data):
    body_data["nframes"] = int(data[0].strip("\r\n"))
    body_data["nbodies"] = []
    body_data["njoints"] = int(data[3].strip("\r\n"))
    body_data["skel_body"] = np.zeros(
        (body_data["nframes"], body_data["njoints"], 3), dtype=np.float32
    )
    body_data["rgb_body"] = np.zeros(
        (body_data["nframes"], body_data["njoints"], 2), dtype=np.float32
    )
    body_data["depth_body"] = np.zeros(
        (body_data["nframes"], body_data["njoints"], 2), dtype=np.float32
    )

    line = 0

    for frame in range(body_data["nframes"]):
        line += 1
        bodycount = int(data[line][:-1])
        print(line, bodycount)
        if bodycount == 0:
            print("Empty Frame")
            continue

        for b in range(body_data["nbodies"]):
            line += 1
            bodyinfo = data[line][:-1].split(' ')
            print(line, bodyinfo)
            line += 1
            for j in range(body_data["njoints"]):
                line += 1
                temp_data = data[line].strip("\r\n").split()
                print(line, temp_data)
                body_data["skel_body"][frame, j] = np.array(
                    temp_data[:3], dtype=np.float32
                )
                body_data["rgb_body"][frame, j] = np.array(
                    temp_data[5:7], dtype=np.float32
                )
                body_data["depth_body"][frame, j] = np.array(
                    temp_data[3:5], dtype=np.float32
                )

    return body_data


if __name__ == "__main__":
    path = "/media/ntfs-data/datasets/ntu/nturgb+d_60_skeletons/"
    datalist = find_all("A001", path)

    action_data = []

    for idx, file in enumerate(datalist):
        print(idx, file)
        body_data = dict()

        with open(path + file, "r") as fr:
            str_data = fr.readlines()

        body_data["setup"] = int(file[1:4])
        body_data["camera"] = int(file[5:8])
        body_data["subject"] = int(file[9:12])
        body_data["replication"] = int(file[13:16])
        body_data["action"] = int(file[17:20])

        body_data = extract_skel_data(str_data, body_data)

        action_data.append(body_data)


    with open("raw_data/A001.pkl", 'wb') as fw:
        pickle.dump(action_data, fw, pickle.HIGHEST_PROTOCOL)
