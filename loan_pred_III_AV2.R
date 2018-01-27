train<-read.csv("C:/Users/Arka/Downloads/trainAVLOAN.csv", stringsAsFactors=T, header = T)

#preprocessing using caret
preprocvalues<- preProcess(train, method = c("knnImpute","center","scale"))
train_processed <-predict(preprocvalues, train)

train_processed$Loan_Status<-ifelse(train_processed$Loan_Status=='N',0,1)
id<-train_processed$Loan_ID
train_processed$Loan_ID<-NULL
 
#creating dummy var for categorical var
str(train_processed)
dmy <- dummyVars(" ~ .", data = train_processed,fullRank = T)
train_transformed <- data.frame(predict(dmy, newdata = train_processed))
str(train_transformed)
train_transformed$Loan_Status<-as.factor(train_transformed$Loan_Status)

#reshuffling the data(splitting)
index <- createDataPartition(train_transformed$Loan_Status, p=0.75, list=FALSE)
trainSet <- train_transformed[ index,]
testSet <- train_transformed[-index,]

#var selection part
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = FALSE)
#finding imp. features by rfe(recursive feature elimination)
outcomeName<-'Loan_Status'
predictors<-names(trainSet)[!names(trainSet) %in% outcomeName]
Loan_Pred_Profile <- rfe(trainSet[,predictors], trainSet[,outcomeName],
                         rfeControl = control)
Loan_Pred_Profile
predictors<-c("Credit_History", "LoanAmount", "Loan_Amount_Term", "ApplicantIncome", "CoapplicantIncome")

#model (caret automatically does bootstrapping)
model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm')

#tuning the parameters
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 5)

modelLookup(model="gbm")
#Creating grid
grid <- expand.grid(n.trees=c(10,20,50,100,500,1000),shrinkage=c(0.01,0.05,0.1,0.5),n.minobsinnode = c(3,5,10),interaction.depth=c(1,5,10))
# training the model again with CV and tuning
model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl=fitControl,tuneGrid=grid)
print(model_gbm)
plot(model_gbm)

#using tune length
model_gbmtunelength<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl=fitControl,tuneLength=10)
print(model_gbm)
plot(model_gbmtunelength)

#Variable Importance
varImp(object=model_gbmtunelength)


#Predictions
predictions1<-predict.train(object=model_gbmtunelength,testSet[,predictors],type="raw")
table(predictions1)
confusionMatrix(predictions1,testSet[,outcomeName])

#using original test set(done alone)
test1<-read.csv("C:/Users/Arka/Downloads/testAVLOAN.csv",header = T,stringsAsFactors = T)
predtest<-data.frame(predict.train(object=model_gbmtunelength,test_transformed,type="raw"))
write.csv(predtest,"file2.csv")
