import numpy as np
from ProbCalculation import roundedProb
from RewardMatrix import rewardMatrix

gamma=0.95

def bellmanEquation(rewardMatrix,gamma,max_iterations,tolerance):

    n= len(rewardMatrix)
    v_old= np.zeros(n)
    for i in range(max_iterations):
        v_new = np.zeros(n)
        for state in range(n):
            action_values= []
            for action in range(n-state):
                expected_future_cost=0
                for demand in range(len(roundedProb)):
                    new_state=max(0,state+action-demand)
                    expected_future_cost+=roundedProb[demand]*v_old[new_state]
                action_value = rewardMatrix[state, action] + gamma * expected_future_cost
                action_values.append(action_value)
            v_new[state]=min(action_values)
        if np.max(np.abs(v_new - v_old)) < tolerance:
            break
        v_old= v_new.copy()
    return v_new

def policy_extraction(V, rewardMatrix, gamma):
    n = len(rewardMatrix)
    policy = np.zeros(n)
    for state in range(n):
        action_values=[]
        for action in range(n-state):
            expected_future_cost=0
            for demand in range(len(roundedProb)):
                new_state = max(0, state + action - demand)
                expected_future_cost += roundedProb[demand] * V[new_state]
            action_value = rewardMatrix[state, action] + gamma * expected_future_cost
            action_values.append(action_value)
        policy[state]=np.argmin(action_values)
    return policy

V= bellmanEquation(rewardMatrix, gamma, 1000, 1e-3)
policy = policy_extraction(V, rewardMatrix, gamma)
#
# print("Optimal ordering policy:")
# print(policy)