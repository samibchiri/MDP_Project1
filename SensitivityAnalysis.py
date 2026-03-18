from BellmanEquation import bellmanEquation,policy_extraction
from ProbCalculation import demandData
from RewardMatrix import rewardMatrixMaker

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


OrderCost=3
HoldingCost=1
gammas=np.linspace(0.95,0.05,10)
shortage_costs=np.linspace(1,20,20)



Sensitivity_matrix=np.zeros((len(gammas),len(shortage_costs)))


for i in range(len(gammas)):
    for j in range(len(shortage_costs)):
        rewardMatrix= rewardMatrixMaker(OrderCost,HoldingCost,shortage_costs[j])
        V=bellmanEquation(rewardMatrix,gammas[i],1000,1e-3)
        policy = policy_extraction(V, rewardMatrix, gammas[i])
        Sensitivity_matrix[i,j]=policy[0]
        print(i,j)

if __name__ == "__main__":
    plt.figure(figsize=(8,6))
    sns.heatmap(
        np.round(Sensitivity_matrix,0),
        #annot=True,         # show numbers in cells
        cmap="coolwarm",    # color map
        xticklabels=[f"{cost}" for cost in shortage_costs],
        yticklabels=[f"{gamma}" for gamma in gammas]
    )
    plt.title("Reward Matrix Heatmap")
    plt.xlabel("ShortageCosts")
    plt.ylabel("Gamma")
    plt.show()


#Lower gamma: less cookies ordered
#Lower shortage costs: less cookies ordered
