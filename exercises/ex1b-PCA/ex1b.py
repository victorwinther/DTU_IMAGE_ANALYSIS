import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition

# Directory containing data and images
in_dir = "data/"

# X-ray image
txt_name = "irisdata.txt"
iris_data = np.loadtxt(in_dir + txt_name, comments="%")
# x is a matrix with 50 rows and 4 columns
x = iris_data[0:50, 0:4]

n_feat = x.shape[1]
n_obs = x.shape[0]
print(f"Number of features: {n_feat} and number of observations: {n_obs}")

sep_l = x[:, 0]
sep_w = x[:, 1]
pet_l = x[:, 2]
pet_w = x[:, 3]

# Use ddof = 1 to make an unbiased estimate
var_sep_l = sep_l.var(ddof=1)
var_sep_w = sep_w.var(ddof=1)
var_pet_l = pet_l.var(ddof=1)
var_pet_w = pet_w.var(ddof=1)

cov_l = sep_l.dot(pet_l)/(n_obs - 1)
cov_w = sep_l.dot(sep_w)/(n_obs -1)

print(f"Sepal length, petal length covariance: {cov_l}. ")
print(f"Sepal length, sepal width covariance: {cov_w}. ")


plt.figure() # Added this to make sure that the figure appear
# Transform the data into a Pandas dataframe
d = pd.DataFrame(x, columns=['Sepal length', 'Sepal width', 'Petal length', 'Petal width'])

p = sns.pairplot(d)
plt.show()

mn = np.mean(x, axis=0)
data = x - mn

c_x = (data.T @ data)/(n_obs - 1)
c_x_np = np.cov(data.T)

np.allclose(c_x, c_x_np)

values, vectors = np.linalg.eig(c_x) # Here c_x is your covariance matrix.

v_norm = values / values.sum() * 100

plt.plot(v_norm)
plt.xlabel("Principal component")
plt.ylabel("Percent explained variance")
plt.ylim([0, 100])

plt.show()

pc_proj = vectors.T.dot(data.T)

pca = decomposition.PCA()
pca.fit(x)
values_pca = pca.explained_variance_
exp_var_ratio = pca.explained_variance_ratio_
vectors_pca = pca.components_

data_transform = pca.transform(x)