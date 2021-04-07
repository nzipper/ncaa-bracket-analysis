data = read.csv("/Users/Sam/Downloads/all_dataset.csv")
data = data[,-1]

data2 = read.csv("/Users/Sam/Downloads/all_teamData.csv")
names(data2)[] = c("Year", "TeamID", "WP", "LP", "Score", "AlsoScore", "FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR", "AST", "TO", "STL", "BLK", "PF","GP")

names(data2)[] = c("Year", "TeamID1", "TeamID2", "Venue", "WP", "LP", "Score", "AlsoScore", "FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR", "AST", "TO", "STL", "BLK", "PF","Outcome")

x = data2[data2$TeamID %in% c(1393,1272),]
x = x[1:2,]

9 - fgm
8-7 pdf
5-6 wpd

names(data)[] = c("Year", "TeamID1", "TeamID2", "Venue",
                   "WPD", "LPD", "ScoreD", "AlsoScoreD",
                   "FGMD", "FGAD", "FGM3D", "FGA3D",
                   "FTMD", "FTAD", "ORD", "DRD",
                   "ASTD", "TOD", "STLD", "BLKD", "PFD","Outcome")
data_mod = data[,c(4,5,7,9:22)]

smp_size <- floor(0.8 * nrow(data_mod))
set.seed(123)
train_ind <- sample(seq_len(nrow(data_mod)), size = smp_size)

train <- data_mod[train_ind, ]
test <- data_mod[-train_ind, ]

recent = data[data$Year > 2015,]

mod1 = glm(Outcome ~ .,data = train, family = "binomial")


preds = predict(mod1, test, type = "response")
plot(test$Outcome, preds)
length(preds)
for(i in 1:length(preds)){
  if(preds[i] > 0.5)
      preds[i] = 1
  else
    preds[i] = 0
}

c = 0
for(i in 1:length(preds)){
  if(preds[i] == test$Outcome[i])
    c = c + 1
}

print(c/length(preds))

ridge2 <- lm.ridge(log(Outcome)~., data=train, lambda=seq(0, 20, 0.001))
plot(ridge2, ylim = c(-100,100))

ridge2

?logisticRidge()

dev.off()

lasso.mod <- glmnet(train[,1:16],train$Outcome,family = "binomial", alpha=1,lambda=10^seq(-2,2,length.out=100)) # alpha=1 => lasso
plot(lasso.mod, xvar = "lambda")

newX <- model.matrix(~.-Outcome,data=test)
newX <- newX[,-1]

s = c(seq(0,1,by = .005))
mse = c()
acc = c()
for(i in 1:201){
  lasso.pred <- predict(lasso.mod,s=s[i],newx=newX,  type = "response")
  corct = 0
  for(j in 1:length(lasso.pred)){
      if(lasso.pred[j] > .5)
        lasso.pred[j] = 1
      else
        lasso.pred[j] = 0
      if(lasso.pred[j] == test$Outcome[j])
          corct = corct + 1
  }
  acc2 = corct/length(lasso.pred)
  acc = c(acc, acc2)
  mse = c(mse,mean( (lasso.pred - test$Outcome)^2 ))
}

minm = which.min(mse)
maxm = which.max(acc)
acc[1]
plot(s,mse, type = "line")
points(s[minm], mse[minm], col = "red")

plot(s,acc, type = "line")
points(s[maxm], acc[maxm], col = "red")

lasso_coef = predict(lasso.mod, type = "coefficients", s = s[1]) # Display coefficients using lambda chosen by CV
lasso_coef

lasso.pred <- predict(lasso.mod,s=.1,newx=newX,  type = "response")
