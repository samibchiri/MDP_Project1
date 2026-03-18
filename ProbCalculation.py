
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
gamma = stats.gamma

data = pd.read_csv("C://Users//Lenovo_Sami//Downloads//inventory_demand_data.csv")
np.set_printoptions(suppress=True)


demandData=[]

for i in (data["demand"]):
    demandData.append((i))

#Fit data to gamma distribution
x=np.linspace(0,np.max(demandData),100)
param = gamma.fit(demandData, floc=0)
pdf_fitted = gamma.pdf(x, *param)
plt.plot(x, pdf_fitted, color='r')

statespaceLength=20
statespaceArray=np.linspace(0,statespaceLength,statespaceLength)

#Compute discrete probabilities based on gamma distribution
prob = gamma.cdf(statespaceArray+0.5, param[0], param[1], param[2]) - gamma.cdf(statespaceArray-0.5, param[0], param[1], param[2])

roundedProb=[]
for i in prob:
    i=i*100
    roundedProb.append(i)

roundedProb=roundedProb/sum(roundedProb)*100
for i in range(len(roundedProb)):
    roundedProb[i]=round(roundedProb[i],2)

roundedProb=roundedProb/100
if __name__ == "__main__":
    plt.bar(range(len(roundedProb)),roundedProb,color="green",width=0.4,alpha=0.8)

    plt.hist(demandData,density=True,color="blue",width=0.4,alpha=0.4)

    plt.show()



