import os
import numpy as np
import pandas as pd


# OUR CONTRIBUTION: FOR KVASIR-INSTRUMENT
def csv2array(root, csv_list):
    img_chunks = []
    label_chunks = []

    for csv_file in csv_list:
        data = pd.read_csv(os.path.join(root, csv_file))
        img_chunks.append(data["image"].to_numpy())
        label_chunks.append(data["mask"].to_numpy())

    img_array = np.concatenate(img_chunks, axis=0)
    label_array = np.concatenate(label_chunks, axis=0)

    return img_array, label_array
