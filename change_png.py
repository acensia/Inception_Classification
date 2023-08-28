import os, glob

# Change the global path that follows yours
l = glob.glob(os.path.join("../../Downloads/fashion/", "*", "*", "*", "*.*"))
for f in l:
    old = os.path.join(f)
    if old.split(".")[-1] != "png":
        os.rename(old, old.replace(old.split(".")[-1], ".png"))
