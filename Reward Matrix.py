import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ProbCalculation import roundedProb,statespaceLength

print(roundedProb)

rewardMatrix=np.zeros((statespaceLength,statespaceLength))

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
plt.figure(figsize=(8,6))
sns.heatmap(
    rewardMatrix,
    annot=True,         # show numbers in cells
    fmt=".1f",          # number format
    cmap="coolwarm",    # color map
    xticklabels=[f"A{a}" for a in range(rewardMatrix.shape[1])],
    yticklabels=[f"S{s}" for s in range(rewardMatrix.shape[0])]
)
plt.title("Reward Matrix Heatmap")
plt.xlabel("Action")
plt.ylabel("State")
plt.show()

#When shortage cost is 5, it is best to make total number of cookies 7,
#but when shortage cost is 50, it is best to order up to 11 cookies