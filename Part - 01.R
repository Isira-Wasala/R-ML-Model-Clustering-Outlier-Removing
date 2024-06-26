# Load necessary libraries

library(readxl)
library(dplyr)
library(ggplot2)
library(factoextra)
library(NbClust)
library(cluster)
library(caret)
library(stats)
library(data.table)
library(fpc)

# Read the Excel file
wine_dataset <- read_excel("C:\\Users\\HP\\Desktop\\20200257_W1962758_Machine Learning CW\\Code\\Whitewine_v6.xlsx")

# Select the first 11 chemical attributes
wine_chemical_attributes <- select(wine_dataset, 1:11)

# Data Preprocessing Part
#Scaling the wine data
scaled_wine_data <- scale(wine_chemical_attributes)

# Print the boxplot for scalled wine data
boxplot(scaled_wine_data)

# Define function to remove outliers using IQR method iteratively until no outliers remain
remove_outliers_iteratively <- function(data, iqr_values) {
  has_outliers <- TRUE
  while (has_outliers) {
    lower_bound <- apply(data, 2, function(x) quantile(x, 0.25) - 1.5 * iqr_values)
    upper_bound <- apply(data, 2, function(x) quantile(x, 0.75) + 1.5 * iqr_values)
    
    # Find indices of outliers of the dataset
    outlier_indices <- apply(data, 1, function(row) any(row < lower_bound | row > upper_bound))
    
    # If there are no more outliers remaining, the loop will stop
    if (!any(outlier_indices)) {
      has_outliers <- FALSE
    } else {
      # If there are outliers, this removes the outliers
      data <- data[!outlier_indices, ]
      # Recalculate IQR values
      iqr_values <- apply(data, 2, IQR)
    }
  }
  
  return(data)
}

# Calculate IQR for each column in the scaled data (scaled_wine_data)
iqr_values <- apply(scaled_wine_data, 2, IQR)

# Remove outliers iteratively from the entire dataset using IQR method
wine_data_without_outliers <- remove_outliers_iteratively(scaled_wine_data, iqr_values)

# Check the dimensions of the dataset before and after removing outliers iteratively
cat("Dimensions of original dataset:", dim(scaled_wine_data), "\n")
cat("Dimensions after iterative outlier removal:", dim(wine_data_without_outliers), "\n")

# Boxplot after removing outliers from the entire dataset
boxplot(wine_data_without_outliers)

# Define function to remove outliers for a specific column using IQR method
remove_outliers_column <- function(data, column_index, iqr_values) {
  lower_bound <- quantile(data[, column_index], 0.25) - 1.5 * iqr_values[column_index]
  upper_bound <- quantile(data[, column_index], 0.75) + 1.5 * iqr_values[column_index]
  
  # Remove outliers
  data <- subset(data, data[, column_index] >= lower_bound & data[, column_index] <= upper_bound)
  
  return(data)
}

# Remove outliers from the second column using IQR method
wine_data_without_outliers <- remove_outliers_column(wine_data_without_outliers, 2, iqr_values)

# Check the dimensions of the dataset before and after removing outliers from the second column
cat("Dimensions after removing outliers from the second column:", dim(wine_data_without_outliers), "\n")
boxplot(wine_data_without_outliers)

# Determining the number of clusters
## Method 1: Elbow Method
set.seed(123)
fviz_nbclust(wine_data_without_outliers, kmeans, method = "wss")

## Method 2: Silhouette Method
fviz_nbclust(wine_data_without_outliers, kmeans, method = "silhouette")

## Method 3: Gap Statistic Method
# Compute the gap statistic using clusGap
gap_statistics <- clusGap(wine_data_without_outliers, FUN = kmeans, K.max = 10, B = 100)

# Visualize the gap statistic to determine the optimal number of clusters
fviz_gap_stat(gap_statistics)

## Method 4: NbClust
nb_cluster <- NbClust(wine_data_without_outliers, distance = "euclidean", min.nc = 2, max.nc = 10, method = "kmeans")
barplot(table(nb_cluster$Best.nc), xlab = "Number of Clusters", ylab = "Votes", main = "NbClust Results")

# K-means Clustering using Optimal Number of Clusters
optimal_clusters <- 2  # Assuming 2 is the optimal number of clusters determined
set.seed(123)
kmeans_clustering_result <- kmeans(wine_data_without_outliers, centers = optimal_clusters, nstart = 25)
fviz_cluster(kmeans_clustering_result, data = wine_data_without_outliers)
# Results from K-means
print(kmeans_clustering_result$centers)
cat("Within cluster sum of squares:", kmeans_clustering_result$tot.withinss, "\n")
cat("Total sum of squares:", sum((wine_data_without_outliers - apply(wine_data_without_outliers, 2, mean))^2), "\n")
cat("Between cluster sum of squares:", kmeans_clustering_result$betweenss, "\n")

# Silhouette Analysis
silhouette_scores <- silhouette(kmeans_clustering_result$cluster, dist(wine_data_without_outliers))
fviz_silhouette(silhouette_scores)

# PCA Analysis
pca_analysis_result <- prcomp(wine_chemical_attributes, scale. = TRUE)
fviz_eig(pca_analysis_result, addlabels = TRUE, ylim = c(0, 100))

# Selecting PCs with cumulative variance > 85%
pca_data <- data.frame(pca_analysis_result$x)
cumulative_variance <- cumsum(pca_analysis_result$sdev^2 / sum(pca_analysis_result$sdev^2)) * 100
selected_pca_data <- pca_data[, cumulative_variance > 85]

# Clustering on PCA-reduced dataset
set.seed(123)
pca_kmeans_clustering_result <- kmeans(selected_pca_data, centers = optimal_clusters, nstart = 25)
print(pca_kmeans_clustering_result$centers)

## Method 1: Elbow Method
optimal_clusters <- 2
set.seed(123)
fviz_nbclust(selected_pca_data, kmeans, method = "wss")

## Method 2: Silhouette Method
fviz_nbclust(selected_pca_data, kmeans, method = "silhouette")

## Method: Gap Statistic Method
# Compute the gap statistic using clusGap
gap_statistics <- clusGap(selected_pca_data, FUN = kmeans, K.max = 10, B = 100)

# Visualize the gap statistic to determine the optimal number of clusters
fviz_gap_stat(gap_statistics)

## Method 4: NbClust
nb_cluster <- NbClust(selected_pca_data, distance = "euclidean", min.nc = 2, max.nc = 10, method = "kmeans")
barplot(table(nb_cluster$Best.nc), xlab = "Number of Clusters", ylab = "Votes", main = "NbClust Results")

fviz_cluster(pca_kmeans_clustering_result, data = selected_pca_data)

# Silhouette plot for PCA-based clustering
pca_silhouette_scores <- silhouette(pca_kmeans_clustering_result$cluster, dist(selected_pca_data))
fviz_silhouette(pca_silhouette_scores)

# Convert PCA-selected data to distance matrix
pca_distance_matrix <- dist(selected_pca_data)

# Calculate Calinski-Harabasz Index
calinski_harabasz_index <- cluster.stats(pca_distance_matrix, pca_kmeans_clustering_result$cluster)$ch

# Print Calinski-Harabasz Index
cat("Calinski-Harabasz Index:", calinski_harabasz_index, "\n")
