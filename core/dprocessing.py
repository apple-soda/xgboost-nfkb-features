import pandas as pd
import numpy as np

def process(features, time_steps, dataset):
    df = pd.DataFrame()
    for feature, time_step in zip(features, time_steps):
        if time_step is not None:
            all_features = [feature + '_' + str(i) for i in range(1, time_step + 1)]
        else:
            all_features = feature
        data_section = dataset[all_features]
        df = pd.concat([df, data_section], axis=1)
    
    return df