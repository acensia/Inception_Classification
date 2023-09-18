import os, glob, json, shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np



def vis(base_path, num_epochs):
    
    labels_acc = {
        "cardigan": [],
        "cargo_pants": [],
        "coat": [],
        "cotton_pants": [],
        "denim_pants": [],
        "hooded": [],
        "jacket": [],
        "knitwear": [],
        "leggings": [],
        "long-sleeved_T-shirt": [],
        "onepiece": [],
        "padding": [],
        "PK-shirt": [],
        "shirt": [],
        "short-sleeved_T-shirt": [],
        "skirt": [],
        "slacks": [],
        "sport_pants": [],
        "sweatshirt": [],
        "zipup": [],
        "short_pants": []
    }



    for i in range(int(num_epochs/5)):
        epoch = 5*i + 5
        txtname = os.path.join(base_path, f"epoch{epoch}", f"totals_{epoch}.txt")
        shutil.copy(txtname, txtname[:-4]+".json")
        txtname = txtname[:-4]+".json"
        with open(txtname, 'r') as j:
            res_dict = json.load(j)
        for l in res_dict:
            labels_acc[l].append(res_dict[l])

    os.makedirs(f"{base_path}/_plotted", exist_ok=True)
    for i, l in enumerate(labels_acc):
        # Plot the lists using Matplotlib
        if i in [0, 4, 8, 12, 16]:
            labels = []
            start = i
            plt.figure(figsize=(10, 6))
        plt.plot(labels_acc[l], label='l')
        labels.append(l)
        if i in [3, 7, 11, 15, 20]:
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title(f'Accuracy change')
            plt.legend(labels)
            plt.ylim(0, 1)
            end = i
            labels.clear()
            # Save the plot to a file using Matplotlib
            plt.savefig(f'{base_path}/_plotted/plot_{start}-{end}.png')

            # Close the plot
            plt.close()

base_path = "./test_result_2"
vis(base_path, 200)