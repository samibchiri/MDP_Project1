import numpy as np
from ProbCalculation import roundedProb,statespaceLength
from RewardMatrix import rewardMatrix
from MarkovianDist import NextCatMatrix
import seaborn as sns
import matplotlib.pyplot as plt

gamma=0.999
OrderCost = 3
HoldingCost = 1
ShortageCost = 5



#print(roundedProb)
roundedProb=np.array(roundedProb)
probCdf=np.zeros(len(roundedProb))
som=0
LowMidHighIndex=[-1,-1]
for i in range(len(roundedProb)):
    som+=roundedProb[i]
    probCdf[i]=som
    if(som>0.01 and LowMidHighIndex[0]==-1):
        LowMidHighIndex[0]=i
    if (som > 0.96 and LowMidHighIndex[1] == -1):
        LowMidHighIndex[1] = i

IndepLowMidHighProb=[0,0,0]

IndepLowMidHighProb[0]=probCdf[LowMidHighIndex[0]]
IndepLowMidHighProb[1]=probCdf[LowMidHighIndex[1]]-probCdf[LowMidHighIndex[0]]
IndepLowMidHighProb[2]=1-IndepLowMidHighProb[0]-IndepLowMidHighProb[1]
#print(IndepLowMidHighProb)
HighMidDiff=LowMidHighIndex[1]-LowMidHighIndex[0]




ScaledProbLowPrevDem=np.zeros(len(roundedProb))
Sums=[0,0,0]

def demandCat(i,LowMidHighIndex):
    if i <= LowMidHighIndex[0]:
        return 0
    elif i<=LowMidHighIndex[1]:
        return 1
    else:
        return 2
for i in range(len(roundedProb)):
    demandIndex=demandCat(i,LowMidHighIndex)
    ProbScaleFactor=(NextCatMatrix[0][demandIndex]/100)/IndepLowMidHighProb[demandIndex]/sum(NextCatMatrix[0])*100
    ScaledProbLowPrevDem[i]=ProbScaleFactor*roundedProb[i]
    Sums[demandIndex] += roundedProb[i]

ScaledProbMidPrevDem=np.zeros(len(roundedProb))
for i in range(len(roundedProb)):
    demandIndex=demandCat(i,LowMidHighIndex)
    ProbScaleFactor=(NextCatMatrix[1][demandIndex]/100)/IndepLowMidHighProb[demandIndex]/sum(NextCatMatrix[1])*100
    ScaledProbMidPrevDem[i]=ProbScaleFactor*roundedProb[i]

ScaledProbHighPrevDem=np.zeros(len(roundedProb))
for i in range(len(roundedProb)):
    demandIndex=demandCat(i,LowMidHighIndex)
    ProbScaleFactor=(NextCatMatrix[2][demandIndex]/100)/IndepLowMidHighProb[demandIndex]/sum(NextCatMatrix[2])*100
    ScaledProbHighPrevDem[i]=ProbScaleFactor*roundedProb[i]

rewardMatrices=[0,0,0]

rewardMatrixLow = np.ones((statespaceLength, statespaceLength)) * 10e6

newroundedProb = [ScaledProbLowPrevDem, ScaledProbMidPrevDem, ScaledProbHighPrevDem]
def catRewardMatrixGen(cat):
    probCat=newroundedProb[cat]
    catRewardMatrix= np.ones((statespaceLength, statespaceLength)) * 10e6
    for state in range(statespaceLength):
        for action in range(statespaceLength - state):
            ExpectedCost = 0
            for demand in range(statespaceLength):
                produced = action
                leftOver = max(0, (state + produced) - demand)
                shortage = max(0, demand - (state + produced))
                addedCost = OrderCost * produced + HoldingCost * leftOver + ShortageCost * shortage
                ExpectedCost += addedCost * probCat[demand]
            catRewardMatrix[state, action] = ExpectedCost
    return catRewardMatrix

rewardMatrices[0]=catRewardMatrixGen(0)
rewardMatrices[1]=catRewardMatrixGen(1)
rewardMatrices[2]=catRewardMatrixGen(2)

def MarkovianBellmanEquation(rewardMatrix,NextCatMatrix,gamma,max_iterations,tolerance):

    n= len(rewardMatrix)
    v_old= np.zeros((n,3))
    for i in range(max_iterations):
        v_new = np.zeros((n,3))
        for state in range(n):
            for cat in range(3):
                rewardMatrix=rewardMatrices[cat]
                action_values= []
                for action in range(n-state):
                    expected_future_cost=0
                    newroundedProb=[ScaledProbLowPrevDem,ScaledProbMidPrevDem,ScaledProbHighPrevDem][cat]
                    for demand in range(len(newroundedProb)):
                        new_state=max(0,state+action-demand)
                        expected_future_cost+=newroundedProb[demand]*v_old[new_state,demandCat(demand,LowMidHighIndex)]
                    action_value = rewardMatrix[state, action] + gamma * expected_future_cost
                    action_values.append(action_value)
                v_new[state,cat]=min(action_values)
        if np.max(np.abs(v_new - v_old)) < tolerance:
            break
        v_old= v_new.copy()
    return v_new

def Markovian_policy_extraction(V, rewardMatrix, gamma):
    n = len(rewardMatrix)
    policy = np.zeros((n,3))
    for state in range(n):
        for cat in range(3):
            if(i!=0):
                rewardMatrix = rewardMatrices[cat]
            action_values = []
            newroundedProb = [ScaledProbLowPrevDem, ScaledProbMidPrevDem, ScaledProbHighPrevDem][cat]
            for action in range(n-state):
                expected_future_cost=0
                for demand in range(len(newroundedProb)):
                    new_state = max(0, state + action - demand)
                    expected_future_cost += newroundedProb[demand] * V[new_state, demandCat(demand,LowMidHighIndex)]
                action_value = rewardMatrix[state, action] + gamma * expected_future_cost
                action_values.append(action_value)
            policy[state,cat]=np.argmin(action_values)
    return policy

V= MarkovianBellmanEquation(rewardMatrix, NextCatMatrix,gamma, 1000, 1e-6)
policy = Markovian_policy_extraction(V, rewardMatrix, gamma)

print(V)
print("Optimal ordering policy:")
print(policy)


print(ScaledProbLowPrevDem)
print(ScaledProbMidPrevDem)
print(ScaledProbHighPrevDem)