import numpy as np

def compute_distances_two_loops(test, train):
  """
  Compute the distance between each test point in X and each training point
  in self.X_train using a nested loop over both the training data and the
  test data.

  Inputs:
  - X: A numpy array of shape (num_test, D) containing test data.

  Returns:
  - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
    is the Euclidean distance between the ith test point and the jth training
    point.
  """
  num_test = test.shape[0]
  num_train = train.shape[0]
  dists = np.zeros((num_test, num_train))
  for i in range(num_test):
      for j in range(num_train):
          #####################################################################
          # TODO:                                                             #
          # Compute the l2 distance between the ith test point and the jth    #
          # training point, and store the result in dists[i, j]. You should   #
          # not use a loop over dimension, nor use np.linalg.norm().          #
          #####################################################################
          # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

          dists[i, j] = np.sum(np.power(test[i] - train[j], 2))

          # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return dists


def compute_distances_one_loop(test, train):
  """
  Compute the distance between each test point in X and each training point
  in self.X_train using a single loop over the test data.

  Input / Output: Same as compute_distances_two_loops
  """
  num_test = test.shape[0]
  num_train = train.shape[0]
  dists = np.zeros((num_test, num_train))
  for i in range(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      # Do not use np.linalg.norm().                                        #
      #######################################################################
      # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

      dists[i, :] = np.sum(np.power(train - test[i], 2), axis=1)

      # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return dists


def compute_distances_no_loops(test, train):
  """
  Compute the distance between each test point in X and each training point
  in self.X_train using no explicit loops.

  Input / Output: Same as compute_distances_two_loops
  """
  num_test = test.shape[0]
  num_train = train.shape[0]
  dists = np.zeros((num_test, num_train))
  #########################################################################
  # TODO:                                                                 #
  # Compute the l2 distance between all test points and all training      #
  # points without using any explicit loops, and store the result in      #
  # dists.                                                                #
  #                                                                       #
  # You should implement this function using only basic array operations; #
  # in particular you should not use functions from scipy,                #
  # nor use np.linalg.norm().                                             #
  #                                                                       #
  # HINT: Try to formulate the l2 distance using matrix multiplication    #
  #       and two broadcast sums.                                         #
  #########################################################################
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

  dists = -2 * np.dot(test, np.transpose(train)) + np.sum(train**2, axis=1) + np.sum(test**2, axis=1)
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return dists


### TESTING ###

dim = 3
n_test = 4
n_train = 8

train_data = np.random.rand(n_train, dim)
test_data = np.random.rand(n_test, dim)
