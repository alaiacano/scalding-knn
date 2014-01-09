library(ggplot2)
x <- read.table("iris_pred.tsv", header=FALSE, sep="\t")
names(x) <- c("id", "class", "class.pred", "sepal.length", "sepal.width")
x$class <- factor(x$class)
x$class.pred <- factor(x$class.pred)
x$train.only <- is.na(x$class.pred)
x$missed <- (x$class != x$class.pred & !x$train.only)

ggplot() +
    geom_point(data=subset(x, !train.only), aes(x=sepal.length, y=sepal.width, color=class, shape=missed), size=4) +
    geom_point(data=subset(x, train.only), aes(x=sepal.length, y=sepal.width, color=class), size=4, shape='x') +
    ggtitle("KNN classification of sepal length/width, k=15\nx's are training data, triangles are misclassified")
