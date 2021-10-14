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

