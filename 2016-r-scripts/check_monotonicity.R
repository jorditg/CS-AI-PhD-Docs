p <- read.csv('kk.csv', header=F)
p$SUM <- rowSums(p[,3:7])
p$p1 <- p$V3/p$SUM
p$p2 <- p$V4/p$SUM
p$p3 <- p$V5/p$SUM
p$p4 <- p$V6/p$SUM
p$p5 <- p$V7/p$SUM
head(p)

monotonic <- function (vector) {
  idxmax <- which.max(vector)
  if(idxmax == 1) {
    if(vector[2] > vector[3] && vector[3] > vector[4] && vector[4] > vector[5]) TRUE else FALSE
  } else if(idxmax == 2) {
    if(vector[3] > vector[4] && vector[4] > vector[5]) TRUE else FALSE
  } else if(idxmax == 3) {
    if(vector[2] > vector[1] && vector[4] > vector[5]) TRUE else FALSE
  } else if(idxmax == 4) {
    if(vector[3] > vector[2] && vector[2] > vector[1]) TRUE else FALSE
  } else if(idxmax == 5) {
    if(vector[4] > vector[3] && vector[3] > vector[2] && vector[2] > vector[1]) TRUE else FALSE
  }
}

p$monotonic <- apply(p[,9:13], 1, monotonic)

