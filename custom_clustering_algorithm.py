import numpy as np
import pandas as pd


class matrix:

    def __init__(self, filename=None):

        # Initialize the array_2d attributes as an empty NumPy array
        self.array_2d = np.array([])

        if filename is not None:

            # Load the data from the CSV file
            self.load_from_csv(filename)

            # Standardise the data
            self.standardise()


    def load_from_csv(self, filename):

        # Read CSV file using Pandas library
        df = pd.read_csv(filename, header=None)

        # Convert Pandas DataFrame to NumPy array
        self.array_2d = df.to_numpy()


    def standardise(self):

        # Check self.array_2d is not empty
        if self.array_2d.size == 0:
            print("Error: self.array_2d is empty")
            return

        # Loop through each column of self.array_2d
        columns = self.array_2d.shape[1]

        # Loop through calculate the fomula values
        for col in range(0, columns):
            column = self.array_2d[:, col]
            mean = np.mean(column)
            max = np.max(column)
            min = np.min(column)

            # Apply Standardization formula
            self.array_2d[:, col] = (column - mean) / (max - min)


    def get_distance(self, other_matrix, row_i):

        # Get Specific_row from Matrix
        row = self.array_2d[row_i]

        # Initialize the List to Store the Distance
        distances = []

        # Loop through Calculate Euclidean Distance between Specific_row to All Other Rows
        for other_row in other_matrix:
            distance = ((row - other_row)**2)
            distances.append([distance])

        # Convert the list of distances to NumPy array (as a matrix with n rows and 1 column)
        return np.array(distances)


    def get_weighted_distance(self, other_matrix, weights, row_i):

        # Get Specific_row from Matrix
        row = self.array_2d[row_i]

        # Initialize the List to Store the Distance
        weighted_distances = []

        # Loop through Calculate Euclidean Distance between Specific_row to All Other Rows
        for other_row in other_matrix:
            distance = np.sum(weights * ((row - other_row)**2))
            weighted_distances.append([distance])

        # Convert the list of weighted distance to NumPy array (as a matrix with n rows and 1 column)
        return np.array(weighted_distances)


    def get_count_frequency(self, S):

        # # Check Cluster Matrix Output column is 1 or not
        if S.shape[1] != 1:
            return 0

        # Flatten S to make it a 1D array for easier processing
        flattened_S = S.flatten()

        # Get the Unique Values and Counts
        unique, counts = np.unique(flattened_S, return_index=True)

        # Create Dictionary Mapping Each Element with its Count
        frequency_dict = dict(zip(unique, counts))

        return frequency_dict



def get_initial_weights(c):

    # Generate m Random Values between 0 and 1
    random_values = np.random.rand(c)

    # Normalize the Random values to make their Sum equal to 1
    normalized_weights = random_values / np.sum(random_values)

    # Reshape Matrix with 1 row and c columns
    return normalized_weights.reshape(1, c)


def get_separation_within(data, centroids, S, K):

    # Get the number of rows (r) and columns (c) in data
    r,c = data.shape

    # Initialize the separation within clusters matrix with 1 row and c columns
    a = np.zeros((1, c))

    # Loop through Cluster
    for k in range(0,K):
        # Loop through Each Row in Data
        for i in range(0,r):

            # Check if the current row i is assigned to cluster k (Uik = 1)
            if S[i,0] == k:

                # Calculate the Euclidean Distance from Row_i to k-th Centroid
                distance = m.get_distance(centroids, i)

                # Accumulate the squared distance
                a += distance[k,0]
    return a


def get_separation_between(data, centroids, S, K):

    # Get the number of rows (r) and columns (c) in data
    r,c = data.shape

    # Initialize the separation between clusters matrix with 1 row and 1 column
    b = np.zeros((1, c))

    # Calculate the overall mean of the dataset for each feature (1 row, c columns)
    overall_mean = np.mean(data, axis=0)

    # Loop through each cluster (k = 0 to K-1)
    for k in range(0,K):

        # Find count of rows are assigned to cluster k (Nk)
        N_k = np.sum(S == k)

        # Loop through each Feature
        for j in range(0,c):

            # Calculate the Euclidean distance
            Distance = (centroids[k,j] - overall_mean[j])**2

            # Accumulate the Separation Value
            b[0,j] += N_k * Distance

    return b


def get_new_weights(data, centroids, weights, S, K):

    # Get the number of rows (r) and columns (c) in data
    r, c = data.shape

    # Calculate the separation within clusters
    a = get_separation_within(data, centroids, S, K)

    # Calculate the separation between clusters
    b = get_separation_between(data, centroids, S, K)

    # Initialize New Weights Matrix with 1 row and c columns
    new_weights = np.zeros((1, c))

    # Calculate the sum of (bv / av) ---> summation(v=0 to c)
    summation_b_divide_a = np.sum(b/a)

    # Loop through Update the each weight
    for j in range(0, c):

        b_divide_a = b[0,j] / a[0,j]

        new_weights[0,j] += 0.5 * (weights[0,j] + (b_divide_a / summation_b_divide_a))

    return new_weights


def get_centroids(data, S, K):

    # Get the number of rows (r) and columns (c) in data
    r, c = data.shape   # (178, 13)

    # Create an Empty centroid Matrix with K rows and c columns
    centroids = np.zeros((K, c))    # Ex: K=4 means np.zeros((4,13))

    # Randomly select the K different rows from data (178 rows)
    centroids_index = np.random.choice(r, K, replace=False)   # [3,50,133,178]

    # Empty centroid Matrix updated with Random K rows matrix
    centroids = data[centroids_index]

    # Initialize Normalized Random Weights with 1 row and 13 columns
    weights = get_initial_weights(c)

    while True:

        # Store old S value before updation with new value in Centroid
        S_old = S.copy()

        # Calculate Weighted Euclidean Distance between row_i and all centroids
        for i in range(0, r):
            distances_to_centroids = m.get_weighted_distance(centroids, weights, i)

            # Find the index of the centroid with the minimum distance
            closest_centroid_index = np.argmin(distances_to_centroids)

            # Update the S matrix with the index of the closest centroid
            S[i,0] = closest_centroid_index

        # Check if S not updated means Clustering perfectly and Break the Loop
        if np.array_equal(S, S_old):
            break

        # Updating the Centroid Position based Recalculation
        for k in range(0, K):

            # Selecting the Assigned Rows for Cluster k ---> Ex k=1 means select all rows of cluster '1' like 60 rows
            assigned_rows = data[S.flatten() == k]

            # If Data Points connected to Centroid means calculate mean to move Centroid position
            if len(assigned_rows) > 0:
                centroids[k] = np.mean(assigned_rows, axis=0)

        # Updating the Weights based Recalculation
        weights = get_new_weights(data, centroids, weights, S, K)

    return S


def get_groups(data, K):

    # Number of Rows from data
    r = data.shape[0]   # 178

    # Initialize Matrix S with r rows and 1 column
    S = np.zeros((r,1))

    # Get Cluster Matrix S and Centroids
    S = get_centroids(data, S, K)

    return S


def run_test():

    global m

    # Initialize object to load CSV input file
    m = matrix('Data.csv')

    # Loop through different Cluster values
    for k in range(2, 11):

        # Iterate multiple times to get Potential outputs
        for i in range(0, 20):

            # Find the Cluster matrix of each row
            S = get_groups(m.array_2d, k)

            # Print the Count Frequency Dict of each Cluster
            print(f'{k}={m.get_count_frequency(S)}')


if __name__ == '__main__':
    try:
        run_test()
    except Exception as e:
        print(f'Error: {e}')
