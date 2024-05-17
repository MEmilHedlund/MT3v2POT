import json
import numpy as np
import glob
import os
from collections import Counter

def get_batch_from_json(infile) -> None:
    
    data = []

    for line in infile:
        data.append(json.loads(line))
    
    #with open(r'jsonfile', 'r') as infile:
    #    for line in infile:
    #        data.append(json.loads(line))

    
    framesize=20
    batches=[]
    all_labels=[]
    unique_ids=[]
    labels_last_step=[]
    unique_label_ids = []

    for k in range(2):
        
        batch=[]
        label=[]
        unique_id=[]

        randominteger=np.random.randint(1,(len(data)-framesize+1))
        counter=0

        basetime=data[randominteger-1][0].get('t')
        
        for n in range(framesize):
            counter += len(data[randominteger+n-1])

        for t in range(framesize):
            row = data[randominteger-1+t]
            for j in range(len(row)):
            ##Batch
                t=round(row[j].get('t')-basetime,3)
                r=row[j].get('r')
                vr=row[j].get('vr')
                phi=row[j].get('phi')
                batch.append([r,vr,phi,t])

            ##Labels
                x=row[j].get('pointcloudx')
                y=row[j].get('pointcloudy')
                vx=row[j].get('vx')
                vy=row[j].get('vy')
                if row[j].get('id') > 0:
                    id=row[j].get('id') 
                    
                else:
                    id=-1
                label.append([x,y,vx,vy,t,id])

            ##Annat
                unique_id.append(id)
            
        batches.append(batch)
        all_labels.append(label) #+counter-len(data[randominteger+n])
        unique_ids.append(unique_id)

        a=[]
        b=[]

        for i in range(len(data[randominteger+n-1])):
            if all_labels[k][len(all_labels[k])-(len(data[randominteger+n-1]))+i][5] > 0:
                a.append(all_labels[k][len(all_labels[k])-(len(data[randominteger+n-1]))+i][0:4])
                b.append(all_labels[k][len(all_labels[k])-(len(data[randominteger+n-1]))+i][5])
            else:
                continue
        labels_last_step.append(a)
        unique_label_ids.append(b)



    return batches, unique_ids, labels_last_step, unique_label_ids    

def open_json(timestamp=str):
    path_pattern = os.getcwd() + "\\" + timestamp
    # Using glob to find matches
    matches = glob.glob(path_pattern)

    # Check if there are any matches and read the first line of the first match
    if matches:
        jsonl_file_path = matches[0]  # Assuming you want to access the first matching file
        # Pattern to match JSONL files
        file_pattern = f"{jsonl_file_path}\\*.jsonl"
        json_file = glob.glob(file_pattern)[0]
        with open(json_file, 'r') as infile:
            batches, unique_ids, labels_last_step, unique_label_ids  = get_batch_from_json(infile)
    else:
        print("No matching .jsonl files found.")
    
    training_data = tuple([np.array(batches[0]), np.array(batches[1])])
    labels = tuple([np.array(labels_last_step[0]), np.array(labels_last_step[1])])
    umids = list([np.array(unique_ids[0]), np.array(unique_ids[1])])
    ulids = list([np.array(unique_label_ids[0]), np.array(unique_label_ids[1])])
    
    for i in range(2):
        id_map = []
        idsm = umids[i]
        idsl = ulids[i]
        unique_ids = np.unique(idsm)
        id_map = {id_val: new_id for new_id, id_val in enumerate(unique_ids)}
        umids[i] = np.array([id_map.get(id_val, -1) if id_val != -1 else -1 for id_val in idsm])
        ulids[i] = np.array([id_map.get(id_val, -1) if id_val != -1 else -1 for id_val in idsl])
    
    unique_measurement_ids = tuple(umids)
    unique_label_ids = tuple(ulids)
    trajectories = []

    return training_data, labels, unique_measurement_ids, unique_label_ids, trajectories

training_data, labels, unique_measurement_ids, _, _ = open_json("*094451")