import enopy as eno
import pandas
from collections import defaultdict

# Parameters
inputpath="../../output/tests/performance_tests/scaling/"
size="1000000"
compressor="partialACA"



for thread in {2,4,8,16}:
    data = defaultdict(list)
    for node in {2,4,8,16,32,64,128}:
        # Open file
        with open(inputpath+"hmat_scaling_"+compressor+"_"+str(node)+"_"+str(thread)+"_"+str(size)+"_"+str(size)+".eno", 'r') as file:
            document=eno.parse(file.read())

        # Read data
        for run in document.sections("Hmatrix"):
            data[str(thread)+"_x"].append(int(run.field("Number_of_MPI_tasks")))
            data[str(thread)+"_y"].append(float(run.field("Blocks_mean")))

    print(data)
    # Write data
    df = pandas.DataFrame(data)
    df.sort_values(str(thread)+"_x").to_csv("hmat_scaling_"+compressor+"_"+str(size)+"_"+str(size)+"_thread_"+str(thread)+".csv",index=False)


data = defaultdict(list)
for node in {2,4,8,16}:
    
    for thread in {2,4,8}:
        # Open file
        with open(inputpath+"hmat_scaling_"+compressor+"_"+str(node)+"_"+str(thread)+"_"+str(size)+"_"+str(size)+".eno", 'r') as file:
            document=eno.parse(file.read())

        # Read data
        for run in document.sections("Hmatrix"):
            data[str(run.field("Number_of_MPI_tasks"))+"_x"].append(thread)
            data[str(run.field("Number_of_MPI_tasks"))+"_y"].append(float(run.field("Blocks_mean")))

# Write data
data_to_plot=defaultdict(list)
data_to_plot['16_x']=data['16_x']
data_to_plot['16_y']=data['16_y']

df = pandas.DataFrame(data_to_plot)
df.sort_values('16_x').to_csv("hmat_scaling_"+compressor+"_"+str(size)+"_"+str(size)+"_mpi_"+'16'+".csv",index=False)

data_to_plot=defaultdict(list)
data_to_plot['32_x']=data['32_x']
data_to_plot['32_y']=data['32_y']

df = pandas.DataFrame(data_to_plot)
df.sort_values('32_x').to_csv("hmat_scaling_"+compressor+"_"+str(size)+"_"+str(size)+"_mpi_"+'32'+".csv",index=False)