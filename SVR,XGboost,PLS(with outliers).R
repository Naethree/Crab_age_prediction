#loading necessary libraries
library(ggplot2)
library(dplyr)
library(reshape2) 
library(mdatools) 
library(RColorBrewer)
library(caret) 
library(tidyr)
library(car)
library(e1071)
library(xgboost)

#importing the data set
crabdata=read.csv("C:/Users/User/Documents/UNI/3rd year/semester 2/ML/Project 1/CrabAgePrediction.csv")
#getting to know the data set
summary(crabdata)
dim(crabdata)
str(crabdata)

#Preprocessing
#checking for NA
colSums(is.na(crabdata))
#check for any 0s
colSums(crabdata==0)
#we can see that height has 2 zero values
#remove records with height=0
crabdata=crabdata[crabdata$Height > 0, ]
#check if height 0 has been removed
colSums(crabdata==0)
#check if any empty string value
colSums(crabdata=="")

#Feature Engineering
#Change Sex column from character to String
crabdata = crabdata %>%
  mutate(Sex = ifelse(Sex == 'F', 'Female', Sex)) %>%
  mutate(Sex = ifelse(Sex == 'M', 'Male', Sex)) %>%
  mutate(Sex = ifelse(Sex == 'I', 'Indeterminate', Sex)) %>%
  mutate(Sex = as.factor(Sex))

str(crabdata)
#Creating a new variable called Age Group by grouping age variable
crabdata = crabdata %>%
  mutate(AgeGroup=case_when(
    Age < 10 ~ "Young",
    Age >= 10 & Age <= 18 ~ "Adult",
    Age > 18 ~ "Old"
  )) %>%
  mutate(AgeGroup=factor(AgeGroup, levels=c("Young", "Adult", "Old")))
summary(crabdata)


#splitting to training and testing 
#random seed for reproduceability
set.seed(123)
indices = sample(1:nrow(crabdata),0.2*nrow(crabdata))
test_data = crabdata[indices, ]
train_data = crabdata[-indices, ]

#SVR
#SVR WITHOUT SCALING
# Fit Support Vector Regression model
svr_model=svm(Age ~ ., data = train_data, 
              type = 'eps-regression', kernel = 'radial')

# Make predictions on the training set
train_predictions <- predict(svr_model, train_data)

# Calculate training MSE for SVR
train_mse_svr <- mean((train_predictions - train_data$Age)^2)

# Make predictions on the test set
test_predictions <- predict(svr_model, test_data)

# Calculate testing MSE for SVR
test_mse_svr <- mean((test_predictions - test_data$Age)^2)

# Print MSE for SVR
print("MSE of SVR:")
print(paste("Training MSE:", train_mse_svr))
print(paste("Testing MSE:", test_mse_svr))


#SCALE AND THEN FIT SVR
# Scale only continuous numeric predictor variables in the training and test datasets
continuous_vars=sapply(train_data, is.numeric)  # Identify numeric variables
train_data_scaled=train_data
train_data_scaled[, continuous_vars]=scale(train_data[, continuous_vars])

test_data_scaled=test_data
test_data_scaled[, continuous_vars]=scale(test_data[, continuous_vars])


# Fit SVR model with scaling
svr_model_scaled <- svm(Age ~ ., data = train_data_scaled, 
                        type = 'eps-regression', kernel = 'radial')

# Make predictions on the training set
train_predictions_scaled <- predict(svr_model_scaled, train_data_scaled)

# Calculate training MSE for SVR scaled
train_mse_svr_scaled <- mean((train_predictions_scaled - train_data_scaled$Age)^2)

# Make predictions on the test set
test_predictions_scaled <- predict(svr_model_scaled, test_data_scaled)

# Calculate testing MSE for SVR scaled
test_mse_svr_scaled <- mean((test_predictions_scaled - test_data_scaled$Age)^2)

# Print MSE for SVR scaled
print("MSE of SVR scaled:")
print(paste("Training MSE:", train_mse_svr_scaled))
print(paste("Testing MSE:", test_mse_svr_scaled))

#XG BOOST
Xtrain_XG=train_data[, -c(1, 9, 10)]  
Xtest_XG=test_data[, -c(1, 9, 10)]

# Fit XGBoost model
xgb_model <- xgboost(data = as.matrix(Xtrain_XG), 
                     label = train_data$Age, 
                     nrounds = 100, 
                     objective = "reg:squarederror")

# Make predictions on the training set
train_predictions_xgb <- predict(xgb_model, as.matrix(Xtrain_XG))

# Calculate training MSE for XGBoost
train_mse_xgb <- mean((train_predictions_xgb - train_data$Age)^2)

# Make predictions on the test set
test_predictions_xgb <- predict(xgb_model, as.matrix(Xtest_XG))

# Calculate testing MSE for XGBoost
test_mse_xgb <- mean((test_predictions_xgb - test_data$Age)^2)

# Print MSE for XGBoost
print("MSE of XGBoost:")
print(paste("Training MSE:", train_mse_xgb))
print(paste("Testing MSE:", test_mse_xgb))


#PLS with scaling
# Create Xtrain and Xtest (predictor variables)
Xtrain=train_data[, -c(1, 9, 10)]  
Xtest=test_data[, -c(1, 9, 10)]

# Create Ytrain and Ytest (response variable)
Ytrain=train_data[, 9, drop = FALSE]
Ytest=test_data[, 9, drop = FALSE]

#Fitting PLR model
Model1=pls(Xtrain, Ytrain, scale = TRUE,cv=1,
           info = "Crab age prediction model")
#Model summary
summary(Model1)
plot(Model1, main = "Model Plot")

# Plot X-scores
plotXScores(Model1, show.labels = TRUE, main = "Scores Plot")
plotXYLoadings(Model1,show.labels =TRUE)

#Model Calibration Results
summary(Model1$res$cal)

#Getting model coefficients for standardize variables
summary(Model1$coeffs,ncomp = 4)

# Predicting on the test set
Pred1 = predict(Model1,Xtest,Ytest)
plot(Pred1)

# Predicting on the test set
Pred1 <- predict(Model1, Xtest, Ytest)

# Extract predicted values from Pred1
predicted_values <- Pred1$y.pred

# Calculate the residuals
residuals <- Ytest$Age - predicted_values

# Square the residuals
squared_residuals <- residuals^2

# Calculate the Mean Squared Error (MSE)
mse <- mean(squared_residuals)

# Print the MSE
print(mse)

# Calculate MAE
mae <- mean(abs(residuals))

# Print the MSE and MAE
print(paste("MSE of PLS scaled:", mse))
print(paste("MAE of PLS scaled:", mae))
