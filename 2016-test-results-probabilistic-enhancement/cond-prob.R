# conditional probalities P(eye1=c|eye2=d)
# with this probabilities we can calculate P(eye1=c ^ eye2=d) as P(eye1=c|eye2=d)*P(eye2=d) or P(eye2=d|eye1=c)*P(eye1=c)
# where P(eye1=c) and P(eye2=d) are given by the model.
# two matrices for all the possibilities of P(eye1=c ^ eye2=d) are calculated
# the max value of the calculated matrix gives us the most probable value for eye1 and eye2 class
data <- read.csv("left_right.csv")

library(mosaic)
cond_prob_lr <- tally(~Left|Right, data=data, format="proportion")
cond_prob_rl <- tally(~Right|Left, data=data, format="proportion")


# probs matrix represents: P(A|B) where A is in columns and B in rows 
#addmargins(cond_prob_lr)
#cond_prob_lr = prop.table(cond_prob_lr,1)
#cond_prob_rl = prop.table(cond_prob_rl,1)

# load the predictions
pred <- read.csv("./512x512/output.csv")
#ordering
pred_order <- pred[order(pred[1]),]
# the two eyes are suposed to be in contigous rows after this ordering

pred_order$alone <- NULL
pred_order$combined <- NULL
pred_order$alone_other <- NULL
for(i in seq(1, nrow(pred_order), by = 2))
{
  prl <- pred_order[i,3:7]
  prob_r = cond_prob_rl%*%t(prl)
  pred_order$alone[i] <- which(prl == max(prl))
  
  prr <- pred_order[i+1,3:7]
  prob_l = cond_prob_lr%*%t(prr)
  pred_order$alone[i+1] <- which(prr == max(prr))
  
  prl <-  prl + prob_l
  prr <-  prr + prob_r
  pred_order$combined[i] <- which(prl == max(prl))
  pred_order$combined[i+1] <- which(prr == max(prr))
  
  pred_order$alone_other[i] <- which(prob_l == max(prob_l))
  pred_order$alone_other[i+1] <- which(prob_r == max(prob_r))
}  

# Kappa with info only of own eye
library(irr)
ratings=data.frame(labels=pred_order$label, res=pred_order$alone)
kappa2(ratings, weight ="squared")
library(caret)
confusionMatrix(data=pred_order$alone, reference=pred_order$label)

# Combined kappa with info of own eye + other eye
ratings=data.frame(labels=pred_order$label, res=pred_order$combined)
kappa2(ratings, weight ="squared")
confusionMatrix(data=pred_order$combined, reference=pred_order$label)

# Kappa with only info of other eye
ratings=data.frame(labels=pred_order$label, res=pred_order$alone_other)
kappa2(ratings, weight ="squared")
confusionMatrix(data=pred_order$alone_other, reference=pred_order$label)


library(GGally)
ggcorr(ratings, palette="RdBu", label=T)
