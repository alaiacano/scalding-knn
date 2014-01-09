library(ggplot2)
x <- read.table("iris_pred.tsv", header=FALSE, sep="\t")
names(x) <- c("id", "class", "class.pred", "sepal.length", "sepal.width")
x$class <- factor(x$class)
x$class.pred <- factor(x$class.pred)
x$missed <- x$class != x$class.pred

ggplot(x, aes(x=sepal.length, y=sepal.width)) +
    geom_point(aes(color=class, shape=missed), size=4) +
    ggtitle("KNN classification of sepal length/width, k=15")
sum(x$missed)