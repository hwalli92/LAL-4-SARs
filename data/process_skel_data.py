import os
import numpy as np


def find_all(action, path):
    result = []
    for file in os.listdir(path):
        if action is file[17:20]:
            result.append(os.path.join(path, file))
    return result


def extract_skel_data(data, body_data):
    body_data["nframes"] = int(data[0].strip("\r\n"))
    body_data["nbodies"] = int(data[1].strip("\r\n"))
    body_data["bodyID"] = data[2].strip("\r\n").split()[0]
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
        for b in range(body_data["nbodies"]):
            line += 2
            for j in range(body_data["njoints"]):
                line += 1
                temp_data = data[line].strip("\r\n").split()
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
    datalist = find_all("A001", "/media/ntfs/datasets/ntu")
    print(datalist)

    exit(0)

    for file in datalist:
        body_data = dict()

        with open("data/" + file, "r") as fr:
            str_data = fr.readlines()

        body_data["setup"] = int(file[1:4])
        body_data["camera"] = int(file[5:8])
        body_data["subject"] = int(file[9:12])
        body_data["replication"] = int(file[13:16])
        body_data["action"] = int(file[17:20])

        body_data = extract_skel_data(str_data, body_data)
