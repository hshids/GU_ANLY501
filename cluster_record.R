library(dplyr)
library(ggplot2)
library(cluster)
library(factoextra)
library(akmeans)
library(scatterplot3d)


#read csv file into the data
df <- read.csv("~/Desktop/CLEAN/cleandata.csv")
head(df)
str(df)
df$total_employed <- as.numeric(df$total_employed)
df$women <- as.numeric(df$women)
df$white <- as.numeric(df$white)
df$black_or_african_american <- as.numeric(df$black_or_african_american)
df$asian <- as.numeric(df$asian)
df$hispanic_or_latino <- as.numeric(df$hispanic_or_latino)
df$total_16_years_and_over <- as.numeric(df$total_16_years_and_over)
df$x16_to_19_years <- as.numeric(df$x16_to_19_years)
df$x20_to_24_years <- as.numeric(df$x20_to_24_years)
df$x25_to_34_years <- as.numeric(df$x25_to_34_years)
df$x35_to_44_years <- as.numeric(df$x35_to_44_years)
df$x45_to_54_years <- as.numeric(df$x45_to_54_years)
df$x55_to_64_years <- as.numeric(df$x55_to_64_years)
df$x65_years_and_over <- as.numeric(df$x65_years_and_over)
df$median_age <- as.numeric(df$median_age)
str(df)
##### Dissimilarity matrix with Euclidean
## Before we can proceed - we need to REMOVE
## all non-numeric columns
df_num <- df[,c(3:17)]
head(df_num)
(dE <- dist(df_num, method = "euclidean"))
(dM <- dist(df_num, method = "manhattan"))
(dMp2 <- dist(df_num, method = "minkowski", p=2))
# Hierarchical clustering using Complete Linkage
(hc_C <- hclust(dM, method = "complete" ))
plot(hc_C)
hc_D <- hclust(dE, method = "ward.D" )
plot(hc_D)
hc_D2 <- hclust(dE, method = "ward.D2" )
plot(hc_D2)
hc_Dm <- hclust(dM, method = "ward.D" )
plot(hc_Dm)
##heat map
heatmap(as.matrix(dE), cexRow = 1, cexCol = 1)
heatmap(as.matrix(dM), cexRow = 1, cexCol = 1)
heatmap(as.matrix(dMp2), cexRow = 1, cexCol = 1)
########## Euclidean k MEANS
km_cluster1 <- kmeans(na.omit(df_num), 2)
km_cluster1$centers
km_cluster2 <- kmeans(na.omit(df_num), 3)
km_cluster2$centers
km_cluster3 <- kmeans(na.omit(df_num), 4)
km_cluster3$centers
#### visualize
#####for k = 2
df_num <- na.omit(df_num)
new_Df <- df_num[,c(1:15)]
fviz_cluster(km_cluster1, data = new_Df,
             ellipse.type = "convex",
             #ellipse.type = "concave",
             palette = "jco",
             axes = c(1, 4), # num axes = num docs (num rows)
             ggtheme = theme_minimal())
#, color=TRUE, shade=TRUE,
#labels=2, lines=0)
df_num$Cluster = km_cluster1$cluster
df_num$Cluster = case_when(
  df_num$Cluster == 1 ~ "good",
  df_num$Cluster == 2 ~ "pass",
  TRUE ~ "fail"
)
df_num$Cluster = as.factor(df_num$Cluster)
head(df_num$Cluster)
#find the 3d cluster
colors = c("#FF0000", "#E69F00", "#56B4E9")
colors = colors[df_num$Cluster]
plot_3d = scatterplot3d(new_Df, color = colors, pch = 16)
legend(plot_3d$xyz.convert(35, 30, 130), legend = levels(df_num$Cluster), col = c("#FF0000", "#E69F00", "#56B4E9"), pch = 16)
########### for k = 3
#df_num <- na.omit(df_num)
fviz_cluster(km_cluster2, data = new_Df,
             ellipse.type = "convex",
             #ellipse.type = "concave",
             palette = "jco",
             axes = c(1, 4), # num axes = num docs (num rows)
             ggtheme = theme_minimal())
#, color=TRUE, shade=TRUE,
#labels=2, lines=0)
df_num$Cluster = km_cluster2$cluster
df_num$Cluster = case_when(
  df_num$Cluster == 1 ~ "good",
  df_num$Cluster == 2 ~ "pass",
  TRUE ~ "fail"
)
df_num$Cluster = as.factor(df_num$Cluster)
head(df_num$Cluster)
#find the 3d cluster
colors = c("#FF0000", "#E69F00", "#56B4E9")
colors = colors[df_num$Cluster]
plot_3d = scatterplot3d(new_Df, color = colors, pch = 16)
legend(plot_3d$xyz.convert(35, 30, 130), legend = levels(df_num$Cluster), col = c("#FF0000", "#E69F00", "#56B4E9"), pch = 16)
############# k = 4
fviz_cluster(km_cluster3, data = new_Df,
             ellipse.type = "convex",
             #ellipse.type = "concave",
             palette = "jco",
             axes = c(1, 4), # num axes = num docs (num rows)
             ggtheme = theme_minimal())
#, color=TRUE, shade=TRUE,
#labels=2, lines=0)
df_num$Cluster = km_cluster3$cluster
df_num$Cluster = case_when(
  df_num$Cluster == 1 ~ "good",
  df_num$Cluster == 2 ~ "pass",
  TRUE ~ "fail"
)
df_num$Cluster = as.factor(df_num$Cluster)
head(df_num$Cluster)
#find the 3d cluster
colors = c("#FF0000", "#E69F00", "#56B4E9")
colors = colors[df_num$Cluster]
plot_3d = scatterplot3d(new_Df, color = colors, pch = 16)
legend(plot_3d$xyz.convert(35, 30, 130), legend = levels(df_num$Cluster), col = c("#FF0000", "#E69F00", "#56B4E9"), pch = 16)
########## optimal cluster

#is.na(new_Df)
wss <- fviz_nbclust(
  as.matrix(new_Df),
  kmeans,
  k.max = 6,
  method = "wss",
  diss = get_dist(as.matrix(new_Df), method = 'euclidean')
)
sil <- fviz_nbclust(new_Df, FUN = hcut, method = "silhouette", k.max = 6) +
  ggtitle("Silhoutte")
plot(wss)
plot(sil)
#################################################################################
######################### extra point prediction ################################
#################################################################################
df1 <- c(12,18,20,15,30,22,50,45,19,25,49,33,42,24,66)
df2 <- c(16,23,35,24,56,54,34,31,65,89,43,23,57,46,75)
df3 <- c(23,45,32,43,78,13,56,78,24,67,78,33,98,64,24)
newdata <- rbind(df1,df2,df3)
predict.kmeans <- function(object, newdata){
  centers <- object$centers
  n_centers <- nrow(centers)
  dist_mat <- as.matrix(dist(rbind(centers, newdata)))
  dist_mat <- dist_mat[-seq(n_centers), seq(n_centers)]
  max.col(-dist_mat)
}
#km_cluster1
test_preds <- predict(km_cluster1, newdata)
test_preds
table(test_preds, 2)

