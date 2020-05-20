import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--inputfile",type=str)
parser.add_argument("--show",type=int,choices=[0,1],default=1)
parser.add_argument("--depth",type=int,default=1)
parser.add_argument("--save",type=str,default="")

args = parser.parse_args()
inputfile = args.inputfile
depth = args.depth

# First Data
data = (pd.read_csv(inputfile,header=None)).T
header = data.iloc[0]
print(header)
data =data[1:]
data.columns = header


# Create Color Map
colormap = plt.get_cmap("Dark2")
norm = colors.Normalize(vmin=min(data[str(depth)]), vmax=max(data[str(depth)]))

# Figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data["x_0"].tolist(), data["x_1"].tolist(), data["x_2"].tolist(),c=colormap(norm(data[str(depth)].tolist())), marker='o')


# Output
if args.show:
    plt.show()


if args.save!="":
    fig.savefig(args.save,bbox_inches = 'tight',
    pad_inches = 0)