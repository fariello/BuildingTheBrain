from sklearn import datasets
import numpy as np
# Load the iris data set, convieniently provided by sklearn
iris = datasets.load_iris()
# We'll only use the 3rd, and 4th columns of the data (the petal length and petal width)
X = iris.data[:,[2,3]]
# Print out the shape of the ndarray so we see that this is a 2-d matrix with 150 rows with two columnns
print("Shape of X: '%s'" %(str(X.shape)))

# Assign y the targets. In this case, there are only thrree flowers
y = iris.target

# So when we print out the unique values of y, we get 0,1,2 representing the three flowers.
print("Unique Values in 'y': %s" %(np.unique(y)))

# If we want to see the target names, we can use iris.target_names
print iris.target_names

# If we want to see the feature names, we can do that too!
print iris.feature_names

# For version 1.8 or later if we want to do the above in one line, we can:
#
# >>> (X,y) = datasets.load_iris(return_X_y=True)
# >>> print("Shape of X: '%s'" %(str(X.shape)))
# >>> print("Unique Values in 'y': %s" %(np.unique(y)))
#
# But we can't specify using only petal length and petal width

# We're going to split the data into a training set and a testing set.
from sklearn.cross_validation import train_test_split

# This convienience function takes the X, y, and a percent of the set
# to be split (test_size, it can also be an int specifying the number
# to split, but that seems less intuitive to me):
X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

