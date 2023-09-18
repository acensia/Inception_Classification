import os, glob

import re, json

def anal_epoch(base_path, pth_epochs):
    
    result_path =f"{base_path}\\epoch{pth_epochs}\\test_result_added_{pth_epochs}.txt"
    correct = result_path+"_correct"
    wrong = result_path+"_wrong"
    correct_labels = {}
    wrong_labels = {}
    total_labels = {}

    # if os.path.isfile(correct):
    #     os.rename(correct, result_path[:-4]+"_correct.txt")
    correct = result_path[:-4]+"_correct.txt"
    # if os.path.isfile(wrong):
    #     os.rename(wrong, result_path[:-4]+"_wrong.txt")
    wrong = result_path[:-4]+"_wrong.txt"

    # Open the file
    with open(correct, 'r') as f:
        # Iterate over each line in the file
        for line in f:
            # Use a regular expression to find "item2" in the line
            match = re.search(r'answer: ([^\n,]+)', line)
            # If a match was found, print it
            if match:
                item2 = match.group(1)
                if item2 not in correct_labels:
                    correct_labels[item2] = 1
                    total_labels[item2] = 1
                else :
                    correct_labels[item2] += 1
                    total_labels[item2] += 1

    # Open the file
    with open(wrong, 'r') as f:
        # Iterate over each line in the file
        for line in f:
            # Use a regular expression to find "item2" in the line
            match = re.search(r'answer: ([^\n,]+)', line)

            # If a match was found, print it
            if match:
                item2 = match.group(1)
                if item2 not in wrong_labels:
                    wrong_labels[item2] = 1
                else :
                    wrong_labels[item2] += 1
                if item2 not in total_labels:
                    total_labels[item2] = 0

    for t in total_labels:
        if total_labels[t] != 0:
            total_labels[t] /= wrong_labels[t] + correct_labels[t]
        total_labels[t] = round(total_labels[t], 2)
    

    with open(f"{base_path}\\epoch{pth_epochs}\\total_correct_{pth_epochs}.txt", 'w') as f:
        json.dump(correct_labels, f, indent=4)

    with open(f"{base_path}\\epoch{pth_epochs}\\total_wrong_{pth_epochs}.txt", 'w') as f:
        json.dump(wrong_labels, f, indent=4)

    with open(f"{base_path}\\epoch{pth_epochs}\\totals_{pth_epochs}.txt", 'w') as f:
        json.dump(total_labels, f, indent=4)



for i in range(1, 41):
    anal_epoch("./test_result", 5*i)