# Custom Clustering Algorithm without Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tFyEmCWP101maJa4ueL7iovtSl-PRSkE?usp=sharing)

**Introduction**

This project focuses on developing a custom clustering algorithm to analyze wine data, providing an alternative to conventional machine learning techniques. The primary objective is to group wine samples based on their characteristics without relying on existing clustering algorithms. By employing mathematical concepts and leveraging NumPy for efficient calculations, we aim to uncover distinct patterns within the data. This approach enhances our understanding of the relationships among different wine samples, offering valuable insights into their grouping.

<br />

**Table of Contents**

1. Key Technologies and Skills
2. Installation
3. Usage
4. Features
5. Contributing
6. License
7. Contact

<br />

**Key Technologies and Skills**
- Python
- Numpy
- Pandas
- Mathematical Modeling
- Algorithm Development

<br />

**Installation**

To run this project, you need to install the following packages:

```python
pip install numpy
pip install pandas
```

<br />

**Usage**

To use this project, follow these steps:

1. Clone the repository: ```git clone https://github.com/gopiashokan/Custom-Clustering-Algorithm-without-Machine-Learning.git```
2. Install the required packages: ```pip install -r requirements.txt```

<br />

**Features**

#### Data Input and Preprocessing:
   - **File Reading:** The project leverages the Pandas library to efficiently read the wine dataset from a CSV file. This approach allows for easy handling and manipulation of data before clustering.

   - **Data Conversion** After loading the dataset, the data is converted into a NumPy array format. This transition facilitates optimized mathematical computations necessary for implementing the custom clustering algorithm.

#### Standardization of Data:
   - **Column-wise Standardization:** Each feature in the dataset undergoes a custom standardization process. This involves calculating the mean, minimum, and maximum values for each column to ensure that all features are normalized effectively.

   - **Bias Prevention:** By standardizing the data, the algorithm mitigates any bias arising from differing scales among features. This normalization is essential for ensuring that all features contribute equally during the distance calculations in the clustering process.

#### Distance Calculation:
   - **Euclidean Distance Measurement:** The algorithm calculates the Euclidean distance between the specified data point and all other points in the dataset. This fundamental step is essential for identifying how closely data points cluster together.

   - **Weighted Distance Calculation:** To enhance clustering accuracy, a weighted distance metric is introduced. This feature considers the importance of each feature in the distance computation, allowing for a more refined clustering process.

#### Cluster Assignment:
   - **Dynamic Centroid Initialization:** The algorithm begins with random selection of centroids from the dataset. This ensures a diverse representation of data points as cluster centers, leading to more effective clustering.

   - **Iterative Cluster Assignment:** Through an iterative process, each data point is assigned to the nearest centroid based on the computed distances. This step is repeated until stable clusters are formed, ensuring that the algorithm converges effectively.

#### Cluster Separation Measurement:
   - **Within-Cluster Distance:** This metric evaluates how closely the data points within a single cluster are grouped around their center (centroid). It measures the average distance of all points in a cluster to the centroid. A lower within-cluster distance indicates that the points are more tightly packed, reflecting better cohesion within the cluster.

   - **Between-Cluster Distance:** This metric assesses the distance between the centers (centroids) of different clusters. It calculates the average distance between the centroids of each cluster and the overall mean of the dataset. A higher between-cluster distance signifies that the clusters are well-separated from each other, which is desirable for effective clustering.

#### Weight Adjustment:
   - **Reassessing Feature Importance:** The algorithm evaluates how effective each feature is in distinguishing between clusters by analyzing the distances both within the clusters and between them. Features that are more influential in separating the clusters receive higher weights, ensuring that their impact is appropriately recognized in the clustering process.

   - **Applying Updated Weights:** The updated weights are then incorporated into the calculations of distance for each data point relative to the clusters. This ongoing adjustment ensures that the clustering process becomes increasingly refined, resulting in more accurate assignments of data points to their respective clusters.

#### Count Frequency:
   - **Cluster Size Analysis:** Once the clustering process is complete, the algorithm counts the number of data points assigned to each cluster based on the user-defined number of clusters (K). This analysis provides insight into how the data is distributed across the various clusters, helping to identify which clusters are more densely populated.

   - **Frequency Distribution Report:** The results are summarized in a clear format, detailing the number of data points in each cluster. This report is essential for evaluating the clustering outcome, as it highlights the effectiveness of the algorithm in grouping similar data points together and shows the balance of data distribution across the specified clusters.


#### References:

   - NumPy: [https://numpy.org/doc/stable/](https://numpy.org/doc/stable/)
   - Pandas: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
   - Task Document: The project was guided by a comprehensive [Task Document](https://github.com/gopiashokan/Custom-Clustering-Algorithm-without-Machine-Learning/blob/main/input/Task%20Document.pdf) that includes detailed instructions, mathematical formulas, and step-by-step guidance for implementing the clustering algorithm. This document served as a crucial resource for understanding the objectives and methodologies required to successfully complete the project.

<br />

**Contributing**

Contributions to this project are welcome! If you encounter any issues or have suggestions for improvements, please feel free to submit a pull request.

<br />

**License**

This project is licensed under the MIT License. Please review the LICENSE file for more details.

<br />

**Contact**

📧 Email: gopiashokankiot@gmail.com 

🌐 LinkedIn: [linkedin.com/in/gopiashokan](https://www.linkedin.com/in/gopiashokan)

For any further questions or inquiries, feel free to reach out. We are happy to assist you with any queries.

