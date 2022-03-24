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
        else:
            df = pd.read_csv(load_dir + name + '_' + sheet + '.csv')
        size = len(df)
        labels = np.full((size, 1), idx) 
        df = np.hstack([df, labels])
        data.append(df)
        
    if (merge == True):
        data = np.vstack(data)
        
    return data

    