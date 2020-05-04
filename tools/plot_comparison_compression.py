import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import argparse

# Input
parser = argparse.ArgumentParser()
parser.add_argument("--inputfile",type=str)
parser.add_argument("--show",type=int,choices=[0,1],default=1)
parser.add_argument("--save",type=str,default="")

args = parser.parse_args()
inputfile = args.inputfile


# Data
data = pd.read_csv(inputfile)
compressions = data.columns[1:] 

# Plot
fig, ax = plt.subplots(1,1)
ax.set_xlabel("Rank")
ax.set_ylabel("Relative error in Frobenius norm")
for compression in compressions:
    ax.semilogy(data["Rank"],data[compression],"x")
ax.legend(compressions)



# Output
if args.show:
    plt.show()


if args.save!="":
    fig.savefig(args.save,bbox_inches = 'tight',
    pad_inches = 0)