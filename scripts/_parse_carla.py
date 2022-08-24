import matplotlib.pyplot as plt
import numpy as np
import re
import json

fil = "/mnt/e/datasets/carla/replays/replay_4.log"
text = open(fil)

states = {

}

with open(fil, mode="r") as bigfile:
    reader = bigfile.read()
    for i,part in enumerate(reader.split("Frame")):
        if i == 0:
            continue
        lines = part.split("\n")
        if "done." in lines:
            continue
        timestep = float(lines[0].split(" ")[3])
        for line in lines:
            words = line.split(" ")
            if len(words) > 1 and words[1] == "Create":
                # create object
                idnum = words[2][:-1]
                poses = words[-3] + words[-2] + words[-1] 
                typ = words[3]
                states[idnum] = {}
                states[idnum]["poses"] = []
                states[idnum]["type"] = typ
                poses = list(eval(poses))
                poses.append(timestep)
                states[idnum]["poses"].append(poses)
            if "Location:" in words:
                idnum = words[3]
                ind = words.index("Location:")
                poses = words[ind + 1] + words[ind + 2] + words[ind + 3]
                poses = list(eval(poses))
                poses.append(timestep)
                states[idnum]["poses"].append(poses)


with open("./states.json", "w") as f:
    json.dump(states, f, indent=4)
