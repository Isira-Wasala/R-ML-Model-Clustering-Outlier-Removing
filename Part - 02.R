# Load libraries for data manipulation, model training, and visualization
library(readxl)
library(nnet)
library(caret)
library(dplyr)
library(ggplot2)  
library(stringr)

# Import the dataset from an Excel file
currency_data <- read_excel("C:\\Users\\HP\\Desktop\\20200257_W1962758_Machine Learning CW\\Code\\ExchangeUSD.xlsx")


# Extract the exchange rate data for USD to EUR
usd_eur_rate <- currency_data %>% pull(`USD/EUR`)

# Split the data into training and testing sets
training_set <- usd_eur_rate[1:400]
testing_set <- usd_eur_rate[401:length(usd_eur_rate)]

# Function to create input and output data for the model
create_input_output <- function(data, lag){
  if (!is.vector(data)) {
    stop("Input data must be a vector.")
  }
  lagged_data <- embed(data, lag + 1)
  input_data <- lagged_data[, -1]
  output_data <- lagged_data[, 1]
  return(list(input = input_data, output = output_data))
}

# Create input-output vectors for different time delays
max_time_delay <- 4
input_output_vectors <- lapply(1:max_time_delay, function(lag) create_input_output(as.vector(training_set), lag))

# Prepare input and output data for model training
training_input <- lapply(input_output_vectors, function(input) input$input)
training_output <- lapply(input_output_vectors, function(input) input$output)

# Normalize the input and output data
normalized_training_input <- lapply(training_input, function(matrix) {
  scaled_matrix <- scale(matrix)
  return(scaled_matrix)
})

normalized_training_output <- lapply(training_output, function(matrix) {
  scaled_matrix <- scale(matrix)
  return(scaled_matrix)
})

# Function to evaluate the performance of the model
evaluate_mlp_model <- function(model, test_data, lag) {
  predicted_values <- predict(model, as.matrix(test_data))
  
  rmse <- sqrt(mean((as.vector(test_data) - predicted_values)^2))
  mae <- mean(abs(as.vector(test_data) - predicted_values))
  mape <- mean(abs(as.vector(test_data) - predicted_values) / as.vector(test_data)) * 100
  smape <- mean(200 * abs(as.vector(test_data) - predicted_values) / (abs(as.vector(test_data)) + abs(predicted_values)))
  
  return(list(RMSE = rmse, MAE = mae, MAPE = mape, sMAPE = smape))
}

# Function to train the MLP model
train_multilayer_perceptron <- function(input, output, units, lag) {
  mlp_model <- nnet(output, input, size = units, linout = TRUE)
  evaluation_metrics <- evaluate_mlp_model(mlp_model, output, lag)
  return(list(model = mlp_model, evaluation = evaluation_metrics))
}

# List to store the trained models and their evaluations
mlp_models_and_evaluations <- list()

# Define the number of hidden units for MLP models
hidden_layer_units <- c(5, 10)

# Train the MLP model for each input vector and number of hidden units
for (i in 1:max_time_delay) {
  training_input <- normalized_training_input[[i]]
  training_output <- normalized_training_output[[i]]
  
  for (units in hidden_layer_units) {
    model_and_evaluation <- train_multilayer_perceptron(training_input, training_output, units, i)
    
    model_name <- paste("Model", sprintf("%02d", ((i-1)*length(hidden_layer_units) + which(hidden_layer_units == units))), 
                        "| Input vector", sprintf("%02d", i))
    mlp_models_and_evaluations[[model_name]] <- model_and_evaluation
  }
}

# Print the evaluation metrics and efficiency for each model
for (model_name in names(mlp_models_and_evaluations)) {
  cat(model_name, "\n")
  cat("Final value:", mlp_models_and_evaluations[[model_name]]$model$value, "\n")
  evaluation <- mlp_models_and_evaluations[[model_name]]$evaluation
  cat("RMSE:", evaluation$RMSE, "\n")
  cat("MAE:", evaluation$MAE, "\n")
  cat("MAPE:", evaluation$MAPE, "%\n")
  cat("sMAPE:", evaluation$sMAPE, "\n")
  
  efficiency <- length(mlp_models_and_evaluations[[model_name]]$model$wts)
  cat("Efficiency (total number of weight parameters):", efficiency, "\n\n")
}

# Identify the best model based on RMSE
best_model_name <- names(mlp_models_and_evaluations)[which.min(sapply(mlp_models_and_evaluations, function(x) x$evaluation$RMSE))]
best_mlp_model <- mlp_models_and_evaluations[[best_model_name]]$model
best_model_input <- input_output_vectors[[as.numeric(str_extract(best_model_name, "\\d+"))]]$input
best_model_output <- input_output_vectors[[as.numeric(str_extract(best_model_name, "\\d+"))]]$output

# Predict the output using the best model
predicted_output_values <- predict(best_mlp_model, as.matrix(best_model_input))

# Plot the predicted output vs. the actual output
plot(predicted_output_values, type = "l", col = "blue", xlab = "Time", ylab = "Exchange Rate")
lines(best_model_output, col = "red")
legend("topright", legend = c("Predicted", "Actual"), col = c("blue", "red"), lty = 1)

print(best_model_name)


#-----------------------------------------------------------------
#----------- one hidden layer begins -----------------------------
#-----------------------------------------------------------------


# Define a new number of hidden units for MLP models
hidden_layer_units <- c(8)

# Train the MLP model for each input vector and new number of hidden units
for (i in 1:max_time_delay) {
  training_input <- normalized_training_input[[i]]
  training_output <- normalized_training_output[[i]]
  
  for (units in hidden_layer_units) {
    model_and_evaluation <- train_multilayer_perceptron(training_input, training_output, units, i)
    
    model_name <- paste("Model", sprintf("%02d", ((i-1)*length(hidden_layer_units) + which(hidden_layer_units == units))), 
                        "| Input vector", sprintf("%02d", i))
    mlp_models_and_evaluations[[model_name]] <- model_and_evaluation
  }
}

# Print the evaluation metrics and efficiency for each model
for (model_name in names(mlp_models_and_evaluations)) {
  cat(model_name, "\n")
  cat("Final value:", mlp_models_and_evaluations[[model_name]]$model$value, "\n")
  evaluation <- mlp_models_and_evaluations[[model_name]]$evaluation
  cat("RMSE:", evaluation$RMSE, "\n")
  cat("MAE:", evaluation$MAE, "\n")
  cat("MAPE:", evaluation$MAPE, "%\n")
  cat("sMAPE:", evaluation$sMAPE, "\n")
  
  efficiency <- length(mlp_models_and_evaluations[[model_name]]$model$wts)
  cat("Efficiency (total number of weight parameters):", efficiency, "\n\n")
}

# Identify the best model based on RMSE
best_model_name <- names(mlp_models_and_evaluations)[which.min(sapply(mlp_models_and_evaluations, function(x) x$evaluation$RMSE))]
best_mlp_model <- mlp_models_and_evaluations[[best_model_name]]$model
best_model_input <- input_output_vectors[[as.numeric(str_extract(best_model_name, "\\d+"))]]$input
best_model_output <- input_output_vectors[[as.numeric(str_extract(best_model_name, "\\d+"))]]$output

# Predict the output using the best model
predicted_output_values <- predict(best_mlp_model, as.matrix(best_model_input))

# Plot the predicted output vs. the actual output
plot(predicted_output_values, type = "l", col = "blue", xlab = "Time", ylab = "Exchange Rate")
lines(best_model_output, col = "red")
legend("topright", legend = c("Predicted", "Actual"), col = c("blue", "red"), lty = 1)

print(best_model_name)
