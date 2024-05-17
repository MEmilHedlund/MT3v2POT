import json
import numpy as np
import os 
import glob

def get_batch(infile) -> None:
    
    data = []

    for line in infile:
        data.append(json.loads(line))
    
    #with open(r'jsonfile', 'r') as infile:
    #    for line in infile:
    #        data.append(json.loads(line))

    basetime=data[0][0].get('t')
    framesize=20
    batch=[]
    label=[]
    unique_id=[]
    batches=[]
    labels=[]
    unique_ids=[]


    for i in range(len(data)):
        row = data[i]
        for j in range(len(row)):
        ##Batch
            t=round(row[j].get('t')-basetime,3)
            r=row[j].get('r')
            vr=row[j].get('vr')
            phi=row[j].get('phi')

            batline=[r,vr,phi,t]
            batch.append(batline)

        ##Labels
            x=row[j].get('pointcloudx')
            y=row[j].get('pointcloudy')
            vx=row[j].get('vx')
            vy=row[j].get('vy')
            if row[j].get('id') > 0:
                lab=row[j].get('id') 
            else:
                lab=-1

            labline=[x,y,vx,vy,t,lab]
            label.append(labline)

        ##Annat
            unique_id.append(lab)

    for k in range(2):
        randominteger=np.random.randint(1,(len(data)-framesize+1))
        counter=0

        for n in range(framesize):
            counter += len(data[randominteger+n])
        
    
        batches.append(batch[randominteger:randominteger+counter])
        labels.append(label[randominteger+counter-len(data[randominteger+n]):randominteger+counter])
        unique_ids.append(unique_id[randominteger:randominteger+counter])

    labels_last_step=[]
    for k in range(len(labels)):
        a=[]
        for n in range(len(labels[k])):
            if labels[k][n][5] > 0:
                a.append(labels[k][n][5])
            else:
                continue
        labels_last_step.append(a)
    
    return batches, labels, unique_ids, labels_last_step
    

def open_json(timestamp=str):
    """
    path_pattern = os.getcwd() + "\\" + timestamp
    path = glob.glob(path_pattern)[0]
    for root, dirs, files in os.walk(path):
        if name in files:
            return print(os.path.join(root, name))
    """
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
            batches, labels, unique_ids, labels_last_step  = get_batch(infile)
    else:
        print("No matching .jsonl files found.")
    
    return batches, labels, unique_ids, labels_last_step


#batches, labels, unique_ids = open_json("*1044")

