import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import numpy as np

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

states = json.load(open("states.json"))

keys = list(states.keys())
ego = np.random.choice([i for i in range(1, len(keys))])
pose_ego = np.array(states[keys[ego]]["poses"])
N = 5
distances = []
plt.figure(figsize=(10,10))
plt.axis('equal')
# plot N agents closest to ego-agent
for key in keys:
    poses = np.array(states[key]["poses"])
    if poses.shape[0] >= pose_ego.shape[0]:
        distances.append(np.linalg.norm(pose_ego[1:, :2] - poses[1:len(pose_ego), :2]))
    else:
        distances.append(1e11)

inds = np.argsort(distances)
for i in range(N):
    ind = inds[i]
    poses = np.array(states[keys[ind]]["poses"])
    x = np.linspace(-1, 1, len(poses) - 1)
    c = np.tan(x)
    plt.scatter(poses[1:, 0], poses[1:, 1], c=c, zorder=N - i)
x = np.linspace(100, 200, len(pose_ego) - 1)
c = np.tan(x)
plt.scatter(pose_ego[1:, 0], pose_ego[1:, 1], color="red", zorder=N * N)
plt.show()
