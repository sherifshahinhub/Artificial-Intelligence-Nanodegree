import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd 
import numpy as np

# import some data to play with
iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
y = iris.target
features_name = iris.feature_names
features = pd.DataFrame(iris.data, columns = features_name)
#0-Compute the cosine similarity matrix for the Iris data set, and comment on the resultant similarity matrix. Also
#use ”imshow” to visualize the resultant similarity matrix
plt.imshow(cosine_similarity(X,X))
plt.show()
X.shape         
m = len(iris.feature_names)
n = len(np.unique(iris.target))
list_of_class_numbers =np.unique(iris.target)

 # 1-Plot the X data for each class alone.
for i in  (list_of_class_numbers):
    plt.figure()
    print (iris.target_names[i])
    for j in range (m):
        plt.subplot(1,4,j+1)
        plt.plot(features[features_name[j]].iloc[np.where(iris.target == i)[0]])        
    plt.show()


#2-Plot the histogram for each class.
for i in list_of_class_numbers:
    print (iris.target_names[i])
    features.iloc[np.where(iris.target == i)[0]].hist(bins=50)
    plt.show()




#3-Use scatter plot to plot every 2 attributes together.

for x_index in range (n):
    for y_index in range(x_index+1,n):
            formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
            plt.figure(figsize=(5, 4))
            plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
            plt.colorbar(ticks=[0, 1, 2], format=formatter)
            plt.xlabel=iris.feature_names[x_index]
            plt.ylabel=iris.feature_names[y_index]
            plt.tight_layout()
            plt.show()
            
#3-Use scatter plot to plot every 3 attributes together.

for x_index in range(m):
    for y_index in range(x_index+1,m):
        for z_index in range(y_index+1,m):
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.scatter(iris.data[:, x_index], iris.data[:, y_index], iris.data[:, z_index],c=iris.target)
                ax.set_xlabel (iris.feature_names[x_index])
                ax.set_ylabel (iris.feature_names[y_index])
                ax.set_zlabel (iris.feature_names[z_index])
                plt.show()

                