import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--inputfile",type=str)
parser.add_argument("--sizeWorld",type=int,default=1)
parser.add_argument("--show",type=int,choices=[0,1],default=1)
parser.add_argument("--save",type=str,default="")

args = parser.parse_args()
inputfile = args.inputfile
sizeWorld = args.sizeWorld

# First Data
size = pd.read_csv(inputfile+"_0.csv",nrows=1,header=None) 
nr = size.iloc[0][0]
nc = size.iloc[0][1]
matrix = np.zeros((nr,nc))

# Figure
fig, axes = plt.subplots(1,1)
axes.xaxis.tick_top()
plt.imshow(matrix)

# Issue: there a shift of one pixel along the y-axis...
shift = axes.transData.transform([(0,0), (1,1)])
shift = shift[1,1] - shift[0,1]  # 1 unit in display coords
shift = 0
# 1/shift  # 1 pixel in display coords

# Loop
for i in range(0,sizeWorld):
    print(i)
    data = pd.read_csv(inputfile+"_"+str(i)+".csv",skiprows=1,header=None) 

    for index, row in data.iterrows():
        matrix[np.ix_(range(row[0],row[0]+row[1]),range(row[2],row[2]+row[3]))]=row[4]
        rect = patches.Rectangle((row[2]-0.5,row[0]-0.5+shift),row[3],row[1],linewidth=0.75,edgecolor='k',facecolor='none')
        axes.add_patch(rect)
        if row[4]>=0 and row[3]/float(nc)>0.05 and row[1]/float(nc)>0.05:
            axes.annotate(row[4],(row[2]+row[3]/2.,row[0]+row[1]/2.),color="white",size=10, va='center', ha='center')





# Colormap
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('YlGn')
new_cmap = truncate_colormap(cmap, 0.4, 1)


# Plot
matrix =np.ma.masked_where(0>matrix,matrix)
new_cmap.set_bad(color="red")
plt.imshow(matrix,cmap=new_cmap,vmin=0, vmax=10)

# Output
if args.show:
    plt.show()


if args.save!="":
    fig.savefig(args.save,bbox_inches = 'tight',
    pad_inches = 0)