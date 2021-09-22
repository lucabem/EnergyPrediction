library(tseries)
library(tsoutliers)
library(ggplot2)
library(tsbox)
library(fpp2)
library(Amelia)
library(mice)
library(dplyr)
library(lubridate)

train <- read.csv(file = 'train_clean.csv')
test  <- read.csv(file = 'test_clean.csv')

train$Date <- as.Date(train$Date)
test$Date  <- as.Date(test$Date)

train    <- train$t
train    <- ts(train,
               frequency = 365)


test    <- test$t
test <-  c(39376.39,  40633.85, 24621.90, test)

autoplot(decompose(train))
modelo <- auto.arima(train,
                     seasonal = T)
pronostico<- forecast(modelo,
                      1,
                      level=95)

train <- c(train, pronostico$mean)
train    <- ts(train,
               frequency = 365)

yhat <- c()

for (i in 1:30){
  print(i)
  modelo <- auto.arima(train,
                       seasonal = T)
  print(modelo)
  
  pronostico<- forecast(modelo,
                        1,
                        level=95)
  train <- c(train, test[i])
  train    <- ts(train,
                 frequency = 365)
  yhat <- c(yhat, 
            pronostico$mean)
}

# Create a first line
plot(1:30, test[1:30], type = "b", frame = FALSE, pch = 19, 
     col = "red", xlab = "x", ylab = "y")
# Add a second line
lines(1:30, yhat, pch = 18, col = "blue", type = "b", lty = 2)
# Add a legend to the plot
legend("topleft", legend=c("Acual", "Pred"),
       col=c("red", "blue"), lty = 1:2, cex=0.8)

