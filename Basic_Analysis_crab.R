#loading necessary libraries
library(ggplot2)
library(dplyr)
library(reshape2) 
library(mdatools) 
library(RColorBrewer)
library(caret) 
library(tidyr)
library(car)


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

# Compute ratios for Shucked.Weight, Viscera.Weight, and Shell.Weight
crabdata <- crabdata %>%
  mutate(Shucked.Weight = Shucked.Weight / Weight,
         Viscera.Weight = Viscera.Weight / Weight,
         Shell.Weight = Shell.Weight / Weight)

# Remove the Weight variable
crabdata <- crabdata %>%
  select(-Weight)
str(crabdata)


#splitting to training and testing 
#random seed for reproduceability
set.seed(123)
indices = sample(1:nrow(crabdata),0.2*nrow(crabdata))
test_data = crabdata[indices, ]
train_data = crabdata[-indices, ]



#PLR
# Create Xtrain and Xtest (predictor variables)
Xtrain=train_data[, -c(1, 8)]  
Xtest=test_data[, -c(1, 8)]

# Create Ytrain and Ytest (response variable)
Ytrain=train_data[, 8, drop = FALSE]
Ytest=test_data[, 8, drop = FALSE]

#Fitting PLR model
Model1=pls(Xtrain, Ytrain, scale = TRUE,cv=1,
           info = "Crab age prediction model")
#Model summary
summary(Model1)
plot(Model1, main = "Model Plot")

# Plot X-scores
plotXScores(Model1, show.labels = TRUE, main = "Scores Plot")
plotXYLoadings(Model1,show.labels =TRUE)

# Set up the distance limits
Modeldist <- setDistanceLimits(Model1, lim.type = "ddrobust")

# Plot XY Residuals with distance limits
plotXYResiduals(Modeldist, show.labels = TRUE, labels = "indices")

# Add lines indicating the boundary for outlier and influential points
abline(h = Modeldist$outlier.limit, col = "red", lty = 2)
abline(h = Modeldist$influence.limit, col = "blue", lty = 2)

# Add legend
legend("topright", legend = c("Outlier Boundary", "Influential Boundary"),
       col = c("red", "blue"), lty = 2)


#Model Calibration Results
summary(Model1$res$cal)

#Getting model coefficients for standardize variables
summary(Model1$coeffs,ncomp = 4)

# Predicting on the test set
Pred1 = predict(Model1,Xtest,Ytest)
plot(Pred1)






##############33
#outlier removal based on box plots
remove_outliers = function(data, column) {
  boxplot_stats = boxplot(data[[column]], plot = FALSE)
  outliers = boxplot_stats$out
  num_outliers = length(outliers)
  filtered_data = data %>% filter(!(!!sym(column) %in% outliers))
  message("Removed ", num_outliers, " outliers from ", column)
  return(filtered_data)
}

numpredictors = c("Length", "Diameter", "Height", "Weight", "Shucked.Weight", "Viscera.Weight", "Shell.Weight")

for (column in numpredictors) {
  train_data_before = nrow(train_data)
  train_data = remove_outliers(train_data, column)
  train_data_after = nrow(train_data)
  num_removed = train_data_before - train_data_after
  message("Column ", column, ": ", num_removed, " rows removed.")
}

#AFTER OUTLIER REMOVAL:
#Create Xnewtrain and Xnewtest (predictor variables)
Xnewtrain=train_data[, -c(1, 9, 10)]  
Xnewtest=test_data[, -c(1, 9, 10)]

#Create Ynewtrain and Ynewtest (response variable)
Ynewtrain=train_data[, 9, drop = FALSE]
Ynewtest=test_data[, 9, drop = FALSE]

#Fitting PLR model
Model2=pls(Xnewtrain, Ynewtrain, scale = TRUE,cv=1,
           info = "Crab age prediction model")
#Model summary
summary(Model2)
plot(Model2, main = "Model Plot")

# Plot X-scores
plotXScores(Model2, show.labels = TRUE, main = "Scores Plot")
plotXYLoadings(Model2, show.labels = TRUE, main = "XY-loadings Plot")

#distance
Modelnew= setDistanceLimits(Model2, lim.type = "ddrobust")
plotXYResiduals(Modelnew, show.labels = TRUE,
                labels = "indices")

#Model Calibration Results
summary(Model2$res$cal)

#Getting model coefficients for standardize variables
summary(Model2$coeffs,ncomp = 4)

# Predicting on the test set
Pred2 = predict(Model2,Xtest,Ytest)
plot(Pred2)

#CHECKING MULTICOLLINEARITY WITH VIF

# Fit a multiple linear regression model
model=lm(train_data$Age ~ .-AgeGroup-Sex, data = train_data)

# Calculate VIF
vif_values=car::vif(model)

# Print the VIF values
print(vif_values)


