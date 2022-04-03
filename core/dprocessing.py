import pandas as pd
import numpy as np

def process_features(features, time_steps, dataset):
    df = pd.DataFrame()
    for feature, time_step in zip(features, time_steps):
        if time_step is not None:
            all_features = [feature + '_' + str(i) for i in range(1, time_step + 1)]
        else:
            all_features = feature
        data_section = dataset[all_features]
        df = pd.concat([df, data_section], axis=1)
    
    return df

def partition_features(features, time_steps, feature_list, bound, dataset):
    present_features = []
    partitioned_list = feature_list[:int(len(feature_list) * bound)]
    partitioned_list = [item for sublist in partitioned_list for item in sublist]
    for feature, time_step in zip(features, time_steps):
        if time_step is not None:
            for i in range(1, time_step):
                if feature + '_' + str(i) in partitioned_list:
                    present_features.append(feature + '_' + str(i))
        else:
            present_features.append(feature)
           
    return dataset[present_features]