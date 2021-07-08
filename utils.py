import matplotlib.pyplot as plt

def plot(traj, obs):
    plt.xlim([np.amin(traj["x"])-1, np.amax(traj["x"])+1])    
    plt.ylim([np.amin(traj["y"])-1, np.amax(traj["y"])+1])    
    plt.scatter(traj["x"], traj["y"], label="Trajectory")    
    print(obs["x"], obs["y"])    
    plt.scatter(obs["x"], obs["y"], s=400, label="Obstacles")    
    plt.legend()    
    plt.show()