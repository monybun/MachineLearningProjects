#########################################################################################################
############################## SMM636 Machine Learning (PRD2 A 2019/20) #################################
##############################            Group Coursework 4            #################################
##############################         Deadline April 17th 2020         #################################
##############################                 GROUP 1                  #################################
#########################################################################################################

## Libraries
library(mclust)       # Modeling package for fitting clustering algorithm   
library(plot3D)       # Plotting package for 3D data visualization
library(ggpubr)
library(factoextra)
library(ggplot2)      # Plotting package for data visualization
library(kernlab)
library(dplyr)
library(fpc)

set.seed(1)

## Load data
customers.csv = read.csv("Mall_Customers.csv", header = TRUE)
customers = customers.csv[,3:5]

## 3D Scatterplot customers (three variables)
x = customers$Age
y = customers$Annual.Income..k..
z = customers$Spending.Score..1.100.

scatter3D(x, y, z, 
          clab = c("Spending", "Score"),
          bty = "g",
          col = ramp.col(c("#33CC99", "#6699FF", "#FF6699")),
          main = "Mall Customer Data",
          pch = 19,
          cex = 1,
          xlab = "Age",
          ylab = "AnnualIncome",
          zlab = "SpendingScore",
          phi = 20,
          theta = 30)

#########################################################################################################
####################################### MODEL-BASED CLUSTERING ##########################################
#########################################################################################################

######################### [1] Multivariate clustering and density estimation ############################
MBclustering = Mclust(customers)
summary(MBclustering)

MBclusterin.dens = densityMclust(customers)
summary(MBclusterin.dens)

## Density plot
plot(MBclusterin.dens, what = "density", data = customers, type = "hdr", points.cex = 0.5)

################################ [2] Retrieve the number of clusters ##################################
### BIC: Optimal number of clusters
plot(MBclustering, what = "BIC")
abline(v = 4, col = "red", lty = 2)

## Model selected
MBclustering.model = MBclustering$modelName

## Covariance matrices
MCclustering.sigma = MBclustering$parameters$variance$sigma

########################## [3] Measure uncertainty and relating probability ###########################
## 2D plot of the uncertaint observations
plot2d.uncert = fviz_mclust(MBclustering, what = "uncertainty")
print(plot2d.uncert)

## Vector of highest uncertainty levels
MBclustering.highUncert = sort(MBclustering$uncertainty, decreasing = TRUE)

## Create a dataframe of the uncertainty level of each observation
MBclustering.uncert <- data.frame(
  id = 1:nrow(customers),
  cluster = MBclustering$classification,
  uncertainty = MBclustering$uncertainty)

## Plotted observations with highest uncertainty level and the belonging cluster
MBclustering.uncert %>%
  group_by(cluster) %>%
  filter(uncertainty > 0.20) %>%
  ggplot(aes(uncertainty, reorder(id, uncertainty))) +
  geom_point(color = "#0000CC") +
  facet_wrap(~ cluster, scales = 'free_y', nrow = 1) +
  ggtitle("Model Based Clustering: Highest Uncertainty", subtitle = "Observations in the respective clusters")

######################################## [4] Modelled Clusters #########################################
## 3D plot of the clusters
scatter3D(x, y, z,
          main = "Clusters",
          colvar = NULL,
          col = c("#33CC99", "#6699FF", "#FF6699", "#FFCC33")[MBclustering$classification],
          pch = 20,
          cex = 1.5,
          xlab = "Age",
          ylab = "AnnualIncome",
          zlab = "SpendingScore",
          bty = "g", colkey = FALSE,
          #type = "h", ticktype = "detailed",
          phi = 20,
          theta = 30)
legend("bottomleft",
       title = "Cluster", 
       c("1", "2", "3", "4"), 
       fill = c("#33CC99", "#6699FF", "#FF6699", "#FFCC33"),
       cex = 1)

## Mean of each variable in each cluster
MBclustering.mean = data.frame(
  Cluster = MBclustering$parameters$mean)

## Visualize the cluster by plotting "Annual Income" and "Spending Score"
plot.AI.SS = ggplot(customers, aes(x = Annual.Income..k.., y = Spending.Score..1.100.)) +
              geom_point(stat = "identity", color = c("#33CC99", "#6699FF", "#FF6699", "#FFCC33")[MBclustering$classification]) +
              scale_color_discrete(name = " ",
                                   breaks = c("1", "2", "3", "4"),
                                   labels = c("Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4")) +
              ggtitle("Customer Segments", subtitle = "Explained by Annual Income")
print(plot.AI.SS)

## Visualize the cluster by plotting "Age" and "Spending Score"
plot.A.SS = ggplot(customers, aes(x = Age, y = Spending.Score..1.100.)) +
            geom_point(stat = "identity", color = c("#33CC99", "#6699FF", "#FF6699", "#FFCC33")[MBclustering$classification]) +
            scale_color_discrete(name = " ",
                                 breaks = c("1", "2", "3", "4"),
                                 labels = c("Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4")) +
            ggtitle("Customer Segments", subtitle = "Explained by Age")
print(plot.A.SS)

######################################### [6] Cluster Validation ########################################
MBclustering.stats = cluster.stats(dist(customers), MBclustering$classification)
MBclustering.val = MBclustering.stats[c("within.cluster.ss", "avg.silwidth")]
MBclustering.val

#########################################################################################################
#################################### PRINCIPAL COMPONENT ANALYSIS #######################################
#########################################################################################################

## Perform principal component analysis on the data
PCclustering = prcomp(customers, scale = TRUE, center = TRUE)
summary(PCclustering)

## Retrieve the first two principle components (PC1 and PC2)
PCclustering$rotation[,1:2]
PCcustomers = data.frame(PCclustering$x[,1:2])

## Visualize the cluster with PC1 and PC2
plot.PCA = ggplot(PCcustomers, aes(x = PC1, y = PC2))+
           geom_point(stat = "identity", color = c("#33CC99", "#6699FF", "#FF6699", "#FFCC33")[MBclustering$classification])+
           scale_color_discrete(name = " ",
                                breaks = c("1", "2", "3", "4"),
                                labels = c("Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"))+
           ggtitle("Customer Segments", subtitle = "Reduced dimensionality with PCA")
print(plot.PCA)
