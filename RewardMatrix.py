import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ProbCalculation import roundedProb,statespaceLength

print(roundedProb)

rewardMatrix=np.ones((statespaceLength,statespaceLength))*10e6

OrderCost=3
HoldingCost=1
ShortageCost=5

for state in range(statespaceLength):
    for action in range(statespaceLength-state):
        ExpectedCost=0
        for demand in range(statespaceLength):
            produced= action
            leftOver=max(0,(state+produced)-demand)
            shortage=max(0,demand-(state+produced))
            addedCost=OrderCost*produced+HoldingCost*leftOver+ShortageCost*shortage
            ExpectedCost+=addedCost*roundedProb[demand]
        rewardMatrix[state,action]=ExpectedCost

print(rewardMatrix)
if __name__ == "__main__":
    mask= rewardMatrix >=1e6
    plt.figure(figsize=(8,6))
    sns.heatmap(
        np.round(rewardMatrix,0),
        annot=True,         # show numbers in cells
        cmap="coolwarm",    # color map
        mask=mask,
        xticklabels=[f"A{a}" for a in range(rewardMatrix.shape[1])],
        yticklabels=[f"S{s}" for s in range(rewardMatrix.shape[0])]
    )
    plt.title("Reward Matrix Heatmap")
    plt.xlabel("Action")
    plt.ylabel("State")
    plt.show()

print(rewardMatrix[0,0])
