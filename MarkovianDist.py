
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ProbCalculation import roundedProb,statespaceLength


from RewardMatrix import ShortageCost

gamma = stats.gamma

data = pd.read_csv("C://Users//Lenovo_Sami//Downloads//inventory_demand_data.csv")
np.set_printoptions(suppress=True)


demandData=[]

countDemands=np.zeros(11)
countLowMidHigh=np.zeros(3)
for i in range(len(data["demand"])):
    demand=data["demand"][i]
    demandData.append((demand))
    if i!=len(data["demand"])-1:
        countDemands[demand]+=1
        if(demand<4):
            countLowMidHigh[0]+=1
        elif(demand<6):
            countLowMidHigh[1] += 1
        else:
            countLowMidHigh[2] += 1

NextStateMatrix=np.zeros((11,11))

prev_i=0
for i in range(len(demandData)-1):
    prev_i=i
    next_i=i+1
    prev_dem=demandData[prev_i]
    next_dem=demandData[next_i]
    NextStateMatrix[prev_dem,next_dem]+=1/countDemands[prev_dem]

if __name__ == "__main__":
    plt.figure(figsize=(8,6))
    sns.heatmap(
        np.round(NextStateMatrix,3),
        #annot=True,         # show numbers in cells
        cmap="coolwarm",    # color map
    )
    plt.title("Reward Matrix Heatmap")
    plt.xlabel("ShortageCosts")
    plt.ylabel("Gamma")
    plt.show()


#Je kan hier niks uit aflezen


#Grouped in low mid and high, based on 33th percentiles

NextCatMatrix=np.zeros((3,3))
for i in range(len(demandData)-1):
    prev_i=i
    next_i=i+1
    prev_dem=demandData[prev_i]
    next_dem=demandData[next_i]
    prev_index=0
    next_index=0
    if prev_dem<4:
        prev_index=0
    elif prev_dem<6:
        prev_index=1
    else:
        prev_index=2

    if next_dem<4:
        next_index=0
    elif next_dem<6:
        next_index=1
    else:
        next_index=2
    #NextCatMatrix[prev_index,next_index]+=1/countLowMidHigh[prev_index]
    NextCatMatrix[prev_index, next_index] += 1

#NextCatMatrix=np.array([[30,0,0],[0,30,0],[0,0,40]])
#NextCatMatrix=np.array([[0,30,0],[0,30,0],[0,40,0]])
#NextCatMatrix=np.array([[30,0,0],[0,30,0],[0,0,40]])
Categories=["Low","Med","High"]
if __name__ == "__main__":
    plt.figure(figsize=(8,6))
    sns.heatmap(
        np.round(NextCatMatrix.T,3),
        annot=True,         # show numbers in cells
        cmap="coolwarm",    # color map
        xticklabels=Categories,
        yticklabels=Categories,
    )
    plt.gca().invert_yaxis()
    plt.title("Reward Matrix Heatmap")
    plt.xlabel("PrevCat")
    plt.ylabel("NextCat")
    plt.show()





