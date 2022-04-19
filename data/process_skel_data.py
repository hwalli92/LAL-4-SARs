import os
import pickle
import numpy as np


def find_all(action, path):
    result = []
    for file in os.listdir(path):
        if file[16:20] == action:
            result.append(file)
    return result


def extract_skel_data(data):

    num_frames = int(data[0].strip("\r\n"))
    num_bodies = []
    bodies_data = []

    for b in range(4):  # max bodies
        body_data = dict()
        body_data["skel_body"] = np.zeros((num_frames, 25, 3), dtype=np.float32)
        body_data["rgb_body"] = np.zeros((num_frames, 25, 2), dtype=np.float32)
        body_data["depth_body"] = np.zeros((num_frames, 25, 2), dtype=np.float32)

        bodies_data.append(body_data)

    line = 0

    for frame in range(num_frames):
        line += 1
        body_count = int(data[line][:-1])

        if body_count == 0:
            continue

        num_bodies.append(body_count)

        for body in range(body_count):
            body_data = bodies_data[body]
            line += 1
            body_info = data[line][:-1].split(" ")

            line += 1
            num_joints = int(data[line].strip("\r\n"))

            for joint in range(num_joints):
                line += 1
                temp_data = data[line].strip("\r\n").split()
                body_data["skel_body"][frame, joint] = np.array(
                    temp_data[:3], dtype=np.float32
                )
                body_data["rgb_body"][frame, joint] = np.array(
                    temp_data[5:7], dtype=np.float32
                )
                body_data["depth_body"][frame, joint] = np.array(
                    temp_data[3:5], dtype=np.float32
                )

            bodies_data[body] = body_data

#    for b in range(4):
#        print(b)
#        if b >= max(num_bodies):
#            bodies_data.pop(b)

    if len(num_bodies) > 0:
        bodies_data = bodies_data[:max(num_bodies)]
        num_bodies = max(num_bodies)
    else:
        num_bodies = 0

    return num_frames, num_bodies, bodies_data


if __name__ == "__main__":
    path = "/media/ntfs-data/datasets/ntu/nturgb+d_60_skeletons/"
    datalist = find_all("A001", path)

    action_data = []

    for idx, file in enumerate(datalist):
        print(idx, file)
        data = dict()

        with open(path + file, "r") as fr:
            str_data = fr.readlines()

        data["setup"] = int(file[1:4])
        data["camera"] = int(file[5:8])
        data["subject"] = int(file[9:12])
        data["replication"] = int(file[13:16])
        data["action"] = int(file[17:20])

        data["nframes"], data["nbodies"], data["bodies_data"] = extract_skel_data(
            str_data
        )

        action_data.append(data)
    with open("raw_data/A001.pkl", "wb") as fw:
        pickle.dump(action_data, fw, pickle.HIGHEST_PROTOCOL)
