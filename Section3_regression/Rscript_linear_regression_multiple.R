
# Set work directory
setwd("D:/Work/Udemy/Machine_Learning_Datasets/Part 2 - Regression/Section 5 - Multiple Linear Regression/")

# import dataset
dataset = read.csv('Salary_Data.csv')

# Preprocess data
dataset$YearsExperience = ifelse(is.na(dataset$YearsExperience), 
                     ave(dataset$YearsExperience, FUN=function(x) mean(x, na.rm=TRUE)),
                     dataset$YearsExperience)

dataset$Salary = ifelse(is.na(dataset$Salary), 
                        ave(dataset$Salary, FUN=function(x) mean(x, na.rm=TRUE)),
                        dataset$Salary)
  # What is done to the data is set by the FUN parameter, it is not optional, 
  # and can apparently be replaced with arbitrary other things than averaging
  # (TODO: NOT SURE WHY THE FUNCTION IS CALLED AVE THEN?)


# Splitting training and test set # (Requires install.packages('caTools'))

library(caTools) # "Activate" the library (can be done manually in packages)
set.seed(123) # Just making sure result will be same as Hadelyn

split = sample.split(dataset$Salary, SplitRatio = 20/30)
          # Note splitRatio here takes percentage of training

training_set = subset(dataset, split==TRUE)
test_set     = subset(dataset, split==FALSE)

# Feature scaling
#training_set[, 2:3] = scale(training_set[, 2:3])
#test_set[, 2:3]     = scale(test_set[, 2:3])
  
# Simple linear regression
regressor = lm(formula = Salary ~ YearsExperience,               
               data = training_set)
    # Type "summary(regressor) to get some info

y_pred = predict(regressor, newdata = test_set)

# Plotting 
library(ggplot2) # Requires install.packages('ggplot2') first

# Training set
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             color = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
             color = 'blue') +
  ggtitle('Traing set') + 
  xlab('Experience in Years') +
  ylab('Salary in $')
  
# Test set
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             color = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Test set') + 
  xlab('Experience in Years') +
  ylab('Salary in $')



















