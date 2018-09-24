
# Set work directory
setwd("D:/Work/Udemy/Machine_Learning_Datasets/Part 2 - Regression/Section 4 - Simple Linear Regression/")

# import dataset
dataset = read.csv('Data_salary.csv')

# Preprocess data
dataset$Age = ifelse(is.na(dataset$Age), 
                     ave(dataset$Age, FUN=function(x) mean(x, na.rm=TRUE)),
                     dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary), 
                        ave(dataset$Salary, FUN=function(x) mean(x, na.rm=TRUE)),
                        dataset$Salary)
  # What is done to the data is set by the FUN parameter, it is not optional, 
  # and can apparently be replaced with arbitrary other things than averaging
  # (TODO: NOT SURE WHY THE FUNCTION IS CALLED AVE THEN?)


# Now convert category data
dataset$Country = factor(dataset$Country, 
                         levels=c('France','Spain','Germany'),
                         labels=c(1,2,3))
dataset$Purchased = factor(dataset$Purchased, 
                           levels=c('No','Yes'),
                           labels=c(0,1))
            # Note that C() is a function that creates a vector..
            # Dummy params are not needed here, since it is already "told" to R 
            # by the factor function

# Splitting training and test set
# ===
# Make sure following library is installed:
# install.packages('caTools')

library(caTools) # "Activate" the library (can be done manually in packages)
set.seed(123) # Just making sure result will be same as Hadelyn

split = sample.split(dataset$Purchased, SplitRatio = 20/30)
          # Note splitRatio here takes percentage of training

training_set = subset(dataset, split==TRUE)
test_set     = subset(dataset, split==FALSE)

# Feature scaling

training_set[, 2:3] = scale(training_set[, 2:3])
test_set[, 2:3]     = scale(test_set[, 2:3])
  







