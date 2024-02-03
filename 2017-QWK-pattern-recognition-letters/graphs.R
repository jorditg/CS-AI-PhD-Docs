best <- read.csv("best.csv")
library(ggplot2)
p <- ggplot(best, aes(bs,qwk_cv)) + facet_grid(. ~ size)
p <- p + aes(shape=factor(loss)) 
p <- p + geom_point(aes(colour=factor(loss)),size=4) 
p <- p + geom_point(colour="grey90", size=1.5)
p <- p + scale_x_log10()
p <- p + ylab("Quadratic Weighted Kappa in test set")
p <- p + xlab("Batch size")
p <- p + labs(shape="Loss", colour="Loss")
p

best$updates <- best$epoch*100/best$bs

tc <- read.csv("training-curves.csv")
p <- ggplot(tc, aes(epoch,qwk)) + facet_grid(. ~ loss)
p <- p + aes(shape=factor(curve)) 
p <- p + geom_line(aes(colour=factor(curve))) + labs(colour="curve")
p <- p + ylab("QWK")
p <- p + xlab("Epoch")

m1 = max(tc[tc$curve=='test' & tc$loss=='log',]$qwk)
m2 = max(tc[tc$curve=='test' & tc$loss=='qwk',]$qwk)
p <- p + geom_hline(aes(yintercept=m1), color="green", linetype="dashed")
p <- p + geom_hline(aes(yintercept=m2), color="blue", linetype="dashed")
p <- p + ylim(0.5, 1.0)
p

