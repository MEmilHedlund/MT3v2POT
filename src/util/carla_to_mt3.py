import json
import numpy as np
import glob
import os
from itertools import count
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random

def rotate_point(x, y, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    x_new = x * np.cos(angle_radians) - y * np.sin(angle_radians)
    y_new = x * np.sin(angle_radians) + y * np.cos(angle_radians)
    
    return x_new, y_new


def extract_json(timestamp=str, n_batches=int, time_steps: int = 20):
    path_pattern = os.getcwd() + "\\" + timestamp
    # Using glob to find matches
    matches = glob.glob(path_pattern)

    # Check if there are any matches and read the first line of the first match
    if matches:
        jsonl_file_path = matches[0]  # Assuming you want to access the first matching file
        # Pattern to match JSONL files
        file_pattern = f"{jsonl_file_path}\\*.jsonl"
        json_file = glob.glob(file_pattern)[0]
    else:
        print("No matching .jsonl files found.")
    
    infile = open(json_file, 'r')
    all_data = [json.loads(line) for line in infile]
    infile.close() # Close file!

    batch_detections = []
    batch_truth = []
    for _ in range(n_batches):

        rand_idx_start = np.random.randint(0, len(all_data) - time_steps + 1)
        basetime = all_data[rand_idx_start][-1].get('t')
        data_set = all_data[rand_idx_start: rand_idx_start + time_steps]

        id_map = {-1: -1}
        id_generator = count(0)

        tset_detections = []
        tset_truth = []
        for time_step_data in data_set:
            tstep_detections = []
            tstep_truth = []
            for detection in time_step_data:
                if detection == time_step_data[0]:
                    ids_truth=time_step_data[0].get('ids_truth')
                    bboxes_truth=time_step_data[0].get('bboxes_truth')

                    tstep_truth=[ids_truth, bboxes_truth]
                # Extract the data directly, assuming these keys always exist
                else:
                    r, vr, phi = detection.get("r"), detection.get("vr"), detection.get("phi")
                    x, y, vx, vy = detection.get('pointcloudx'), detection.get('pointcloudy'), detection.get('vx'), detection.get('vy')
                    t = round(detection.get("t") - basetime, 3)
                    id = detection.get('id', -1)
                    
                    if id > -1 and id not in id_map:
                        id_map[id] = next(id_generator)
                    mapped_id = id_map.get(id, -1)

                    tstep_detections.append([mapped_id, r, vr, phi, x, y, vx, vy, t])

            np.random.shuffle(tstep_detections) # Shuffles detections in each time step
            tset_detections.append(np.array(tstep_detections))
            tset_truth.append(tstep_truth)
            

        batch_detections.append(tset_detections)
        batch_truth.append(tset_truth)

    
    return batch_detections, batch_truth

def unpack_data_extended(batch, truth):

    # Training Data
    training = tuple()
    trajectories = tuple()
    umids = tuple()
    for train_set in batch:
        training_set = []
        trajectories_set= {}
        umids_set = []

        for tstep in train_set:
            for detection in tstep: 
                id, r, vr, phi, x, y, vx, vy, t = detection[[0, 1, 2, 3, 4, 5, 6, 7, 8]]

                # training
                training_set.append([r, vr, phi, t])
                # trajectories
                id = int(id)
                if id >-1:
                    # values are taken from radar measurments
                    
                    yaw = -90
                    xx = r*np.cos(phi)
                    yy = r*np.sin(phi)
                    vxx = vr*np.cos(phi)
                    vyy= vr*np.sin(phi)
                    """
                    vxx = vx*np.cos(np.deg2rad(yaw)) - vy*np.sin(np.deg2rad(yaw))
                    vyy = vy*np.cos(np.deg2rad(yaw)) + vx*np.sin(np.deg2rad(yaw))
                    """
                    trajectories_set.setdefault(id, []).append([xx, yy, vxx, vyy, t])
                # unique_measurment_ids
                umids_set.append(id)
                
        for id in trajectories_set:
            trajectories_set[id] = np.array(trajectories_set[id]) # have to convert to ndarray

        #trajectories += (dict(sorted(trajectories_set.items())),) # might be a bad idea to order objects 
        trajectories += (trajectories_set,)
        training += (np.array(training_set),)
        umids += (np.array(umids_set),)

    # Labels
    labels = tuple()
    ulids = tuple()
    for truth_set in truth:
        label_set = []
        ulids_set = []
        flat_lab = []

        ### MUltiple labels per set
        #for obj in truth_set:
        #    ulids_set.append(obj[0])
        #    for id in obj[0]:
        #        label_set.append(obj[1].get(str(id)))
        
        #SIngle labels per set
        obj = truth_set[-1]
        for id in obj[0]:
            ulids_set.append(id)
            label_set.append(np.array(obj[1].get(str(id))[0] + [obj[1].get(str(id))[-1]]))
        
        for arr in label_set: flat_lab.append(arr.flatten())

        labels += (np.array(flat_lab),) #for arr in labels[0]: print(arr.flatten()) works for making the individual vehicle flattened values
        ulids += (np.array(ulids_set),)

    return training, labels, umids, ulids, trajectories, None

def unpack_data_point(batch, truth):

    # Training Data
    training = tuple()
    trajectories = tuple()
    umids = tuple()
    for train_set in batch:
        training_set = []
        trajectories_set= {}
        umids_set = []

        for tstep in train_set:
            for detection in tstep: 
                id, r, vr, phi, x, y, vx, vy, t = detection[[0, 1, 2, 3, 4, 5, 6, 7, 8]]

                # training
                training_set.append([r, vr, phi, t])
                # trajectories
                id = int(id)
                if id >-1:
                  trajectories_set.setdefault(id, []).append([x, y, vx, vy, t])
                # unique_measurment_ids
                umids_set.append(id)
                
        for id in trajectories_set:
            trajectories_set[id] = np.array(trajectories_set[id]) # have to convert to ndarray

        #trajectories += (dict(sorted(trajectories_set.items())),) # might be a bad idea to sort
        trajectories += (trajectories_set,)
        training += (np.array(training_set),)
        umids += (np.array(umids_set),)

    # Labels
    labels = tuple()
    ulids = tuple()
    for truth_set in truth:
        label_set = []
        ulids_set = []
        flat_lab = []
        center_lab=[]

        ### MUltiple labels per set
        #for obj in truth_set:
        #    ulids_set.append(obj[0])
        #    for id in obj[0]:
        #        label_set.append(obj[1].get(str(id)))
        
        #SIngle labels per set
        obj = truth_set[-1]
        for id in obj[0]:
            ulids_set.append(id)
            label_set.append(np.array(obj[1].get(str(id))[0] + [obj[1].get(str(id))[-1]]))
        
        for arr in label_set: flat_lab.append(arr.flatten())

        for arr in flat_lab: center_lab.append([sum(arr[0:7:2])/4,sum(arr[1:8:2])/4,arr[-2],arr[-1]])

        labels += (np.array(center_lab),) #for arr in labels[0]: print(arr.flatten()) works for making the individual vehicle flattened values
        ulids += (np.array(ulids_set),)

    return training, labels, umids, ulids, trajectories, None

data_detections, data_truth = extract_json("*094451", n_batches=2, time_steps=20)
#training_data, labels, unique_measurement_ids, unique_label_ids, trajectories, _= unpack_data_extended(data_detections, data_truth)
training_data, labels, unique_measurement_ids, unique_label_ids, trajectories, _= unpack_data_point(data_detections, data_truth)