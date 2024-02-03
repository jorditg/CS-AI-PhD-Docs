# Graph 1
best <- read.csv("retine-best.csv")
best$loss <- as.factor(best$loss)
levels(best$loss)[levels(best$loss)=="log"] <- "log-loss"
levels(best$loss)[levels(best$loss)=="qwk"] <- "qwk-loss"
library(ggplot2)
library(latex2exp)
p <- ggplot(best, aes(bs,qwk_cv)) + facet_grid(. ~ size)
p <- p + aes(shape=factor(loss)) 
p <- p + geom_point(aes(colour=factor(loss)),size=3) 
p <- p + geom_point(colour="grey90", size=1.5)
p <- p + scale_x_log10()
p <- p + ylab(TeX("$\\kappa_{val}$"))
p <- p + xlab("Batch size")
p <- p + labs(shape="Loss", colour="Loss")
p


# Graph 2
best$updates <- best$epoch*100/best$bs
tc <- read.csv("training-curves.csv")
loss_names <- c(
  `log` = "log-loss",
  `qwk` = "qwk-loss"
)
p <- ggplot(tc, aes(epoch,qwk)) + facet_grid(. ~ loss, labeller=as_labeller(loss_names))
p <- p + aes(shape=factor(curve)) 
p <- p + geom_line(aes(colour=factor(curve))) + labs(colour="curve")
p <- p + ylab("QWK")
p <- p + xlab("Epoch")
m1 = max(tc[tc$curve=='val' & tc$loss=='log',]$qwk)
m2 = max(tc[tc$curve=='val' & tc$loss=='qwk',]$qwk)
p <- p + geom_hline(aes(yintercept=m1), color="green", linetype="dashed")
p <- p + geom_hline(aes(yintercept=m2), color="blue", linetype="dashed")
#p <- p + ylim(0.5, 1.0)
p


# Graph 3
library(ggplot2)
library(latex2exp)
#crowdflower
N <- 1000
d1 <- rnorm(N, 0.4679, 0.0498/1.96)
d2 <- rnorm(N, 0.4996, 0.0426/1.96)
data1 <- data.frame(rep('log-loss',N), d1)
colnames(data1) <- c('loss', 'qwk')
data2 <- data.frame(rep('qwk-loss',N), d2)
colnames(data2) <- c('loss', 'qwk')
data <- rbind(data1, data2)
data$loss <- factor(data$loss)
plot1 <- ggplot(data, aes(loss, qwk), colour=loss) + guides(fill=FALSE)
plot1 <- plot1 + geom_boxplot(outlier.shape = NA, aes( fill = loss)) + xlab("Case 1") + ylab(TeX("$\\kappa_{test}$"))
#prudential
N <- 1000
d1 <- rnorm(N, 0.5618, 0.018/1.96)
d2 <- rnorm(N, 0.6175, 0.016/1.96)
data1 <- data.frame(rep('log-loss',N), d1)
colnames(data1) <- c('loss', 'qwk')
data2 <- data.frame(rep('qwk-loss',N), d2)
colnames(data2) <- c('loss', 'qwk')
data <- rbind(data1, data2)
data$loss <- factor(data$loss)
plot2 <- ggplot(data, aes(loss, qwk), colour=loss) + guides(fill=FALSE)
plot2 <- plot2 + geom_boxplot(outlier.shape = NA, aes( fill = loss)) + xlab("Case 2") + ylab(TeX("$\\kappa_{test}$"))
#retine
N <- 1000
d1 <- rnorm(N, 0.686, 0.008/1.96)
d2 <- rnorm(N, 0.740, 0.006/1.96)
data1 <- data.frame(rep('log-loss',N), d1)
colnames(data1) <- c('loss', 'qwk')
data2 <- data.frame(rep('qwk-loss',N), d2)
colnames(data2) <- c('loss', 'qwk')
data <- rbind(data1, data2)
data$loss <- factor(data$loss)
plot3 <- ggplot(data, aes(loss, qwk), colour=loss) + guides(fill=FALSE)
plot3 <- plot3 + geom_boxplot(outlier.shape = NA, aes( fill = loss)) + xlab("Case 3") + ylab(TeX("$\\kappa_{test}$"))
library(gridExtra)
grid.arrange(plot1, plot2, plot3, ncol=3, nrow =1)

# Graph 4
crowdflower <- read.csv("crowdflower.csv")
crowdflower$loss <- as.factor(crowdflower$loss)
levels(crowdflower$loss)[levels(crowdflower$loss)=="log"] <- "log-loss"
levels(crowdflower$loss)[levels(crowdflower$loss)=="qwk"] <- "qwk-loss"
library(ggplot2)
library(latex2exp)
pr_agg <- aggregate(qwk ~ bs + loss, data = crowdflower, max)
p <- ggplot(pr_agg, aes(bs,qwk))
p <- p + aes(shape=factor(loss)) 
p <- p + geom_point(aes(colour=factor(loss)),size=3) 
p <- p + geom_point(colour="grey90", size=1.5)
p <- p + scale_x_log10()
p <- p + ylab(TeX("$\\kappa_{val}$"))
p <- p + xlab("Batch size")
p <- p + labs(shape="Loss", colour="Loss")
p

# Graph 5
prudential <- read.csv("prudential.csv")
prudential$loss <- as.factor(prudential$loss)
levels(prudential$loss)[levels(prudential$loss)=="log"] <- "log-loss"
levels(prudential$loss)[levels(prudential$loss)=="qwk"] <- "qwk-loss"
library(ggplot2)
pr_agg <- aggregate(qwk ~ bs + loss, data = prudential, max)
p <- ggplot(pr_agg, aes(bs,qwk))
p <- p + aes(shape=factor(loss)) 
p <- p + geom_point(aes(colour=factor(loss)),size=3) 
p <- p + geom_point(colour="grey90", size=1.5)
p <- p + scale_x_log10()
p <- p + ylab(TeX("$\\kappa_{val}$"))
p <- p + xlab("Batch size")
p <- p + labs(shape="Loss", colour="Loss")
p


# histogram
require(ggplot2)
library(plyr)
conf_qwk <- read.csv("test-unique-confusion-512-best-qwk.csv")
conf_qwk$real <- as.factor(conf_qwk$real)
conf_qwk$predicted <- as.factor(conf_qwk$predicted)
conf_qwk <- ddply(conf_qwk, .(real), transform, percent = value/sum(value)*100)
plot1 <- ggplot(conf_qwk, aes(x = predicted, y = percent, fill = predicted, width=0.75)) + 
  labs(y = "Prediction [ % ] (qwk-loss)", x = NULL, fill = NULL) +
  geom_bar(stat = "identity") +
  facet_wrap(~real, ncol = 5) +
  guides(fill=FALSE) 
#  theme_bw() + theme( strip.background  = element_blank(),
#                      panel.grid.major = element_line(colour = "grey80"),
#                      panel.border = element_blank(),
#                      axis.ticks = element_line(size = 0),
#                      panel.grid.minor.y = element_blank(),
#                      panel.grid.major.y = element_blank() ) +
#  theme(legend.position="bottom") +
#  scale_fill_brewer(palette="Set2")

conf_log <- read.csv("test-unique-confusion-512-best-log.csv")
conf_log$real <- as.factor(conf_qwk$real)
conf_log$predicted <- as.factor(conf_qwk$predicted)
conf_log <- ddply(conf_log, .(real), transform, percent = value/sum(value)*100)
plot2 <- ggplot(conf_log, aes(x = predicted, y = percent, fill = predicted, width=0.75)) + 
  labs(y = "Prediction [ % ] (log-loss)", x = NULL, fill = NULL) +
  geom_bar(stat = "identity") +
  facet_wrap(~real, ncol = 5) +
  guides(fill=FALSE) 
#  theme_bw() + theme( strip.background  = element_blank(),
#                    panel.grid.major = element_line(colour = "grey80"),
#                      panel.border = element_blank(),
#                      axis.ticks = element_line(size = 0),
#                      panel.grid.minor.y = element_blank(),
#                      panel.grid.major.y = element_blank() ) +
#  theme(legend.position="bottom") +
#  scale_fill_brewer(palette="Set2")


library(gridExtra)
grid.arrange(plot1, plot2, ncol=1, nrow =2)
