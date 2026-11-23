from matplotlib import pyplot as plt
from matplotlib import pylab as pyl
import csv
import os
import pandas as pd
import numpy as np
from matplotlib.colors import Normalize

def analyse_df(dataframe:pd.DataFrame)->pd.DataFrame:
    # average over trials per simulation
    groups = dataframe.groupby(["height", "width","fill_chance","update_chance","survival_chance","wrap","generations"])
    grouped = groups.agg(
        {
            "time": "mean",
            "alive":"mean",
            "inactive": "mean",
        }
    ).reset_index()
    return grouped
         
cmap_name = "viridis"
colormap = pyl.cm.get_cmap("viridis")


current_directory = os.getcwd()

data_sets = [filename  for filename in os.listdir(current_directory) if (filename[-4::] == ".csv")]

for filename in data_sets:
    data = analyse_df(pd.read_csv(filename, header="infer"))
    data = data.round(4)
    
    # arbitrary start for initial density
    data = data[data["fill_chance"] == 0.5]
    
    #data = data[data["update_chance"] == 1.0]
    #data = data[data["survival_chance"] == 0.5]
    
    data["area"] = data["width"]*data["height"]
    data = data.reset_index()
    
    plt.figure(0)
    fig1, ax1 = plt.subplots()
    fig1.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1), cmap=colormap ), ax=ax1, label = "Chance to update each cell")
    
    #ax1.scatter(data["area"],data["time"]*2,c=data["update_chance"],cmap=colormap)
    grouped = [group for _, group in data.groupby("update_chance")]
    colors = colormap(np.linspace(0,1, len(grouped)))
    for i,line in enumerate(grouped):
        ax1.plot(line["area"],line["time"],color=colors[i])
    ax1.set_xlabel("Grid size")
    ax1.set_ylabel("Time per generation [ms]")
    
    
    plt.figure(1)
    fig2, ax2 = plt.subplots()
    fig2.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1), cmap=cmap_name), ax=ax2, label = "Chance to update each cell")
    ax2.scatter(data["area"],data["alive"]/data["area"],c=data["update_chance"],cmap=colormap)
    ax2.set_xlabel("Grid size")
    ax2.set_ylabel("Surviving Cells%")
    
        
    plt.show()
    
    
    
    
