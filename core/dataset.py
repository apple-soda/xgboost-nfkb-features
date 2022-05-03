import numpy as np
import pandas as pd

def Data(load_dir, names, sheet, merge=False, numpy=False):
    data = []
    for idx, name in enumerate(names):
        """
        convert excel sheet with all metrics to csv file
        enumerate to add categorical labels to each matrix
        concatenate labels and features
        """
        if (numpy == True):
            df = pd.read_csv(load_dir + name + '_' + sheet + '.csv').to_numpy()
            size = len(df)
            labels = np.full((size, 1), idx) 
            df = np.hstack([df, labels])
            data.append(df)
        
        else:
            df = pd.read_csv(load_dir + name + '_' + sheet + '.csv')
            size = len(df)
            labels = np.full((size, 1), idx)
            labels = pd.DataFrame(labels)
            df = pd.concat([df, labels], axis=1)
            data.append(df)
        
    if (merge == True and numpy == False):
        data = pd.concat(data)
    if (merge == True and numpy == True):
        data = np.vstack(data)
        
    return data

def PolarData(load_dir, names, polar_states, merge=False, numpy=False):
    data = []
    for i in names: # CpG_am
        for num, state in enumerate(polar_states):
            if state == '':
                x = pd.read_csv(load_dir + i + '.csv')
            else:
                x = pd.read_csv(load_dir + i + '_' + state + '.csv')
            label = np.full((len(x), 1), num)
            label = pd.DataFrame(label)
            x = pd.concat([x, label], axis=1) # append to end of dataframe
            data.append(x)
    
    if merge == True:
        data = pd.concat(data) # append along axis=0
    
    return data
            
    