# ST3189 Coursework Project
# # Student SRN: 230713279
# Dataset: The World Development Indicators (WDI)
# World Bank Open Data
# Link: https://databank.worldbank.org/source/world-development-indicators


# ============================================
# 1. SETUP AND LIBRARIES
# ============================================
library(dplyr)
library(tidyr)
library(lmtest)
library(car)
library(caret)
library(sandwich)
library(corrplot)
library(glmnet)
library(ggplot2)
library(factoextra)
library(cluster)
library(reshape2)
library(randomForest)
library(gt)
library(Metrics)
library(tibble)
library(pROC)
library(PRROC)
library(mlr3)
library(mlr3learners)
library(mlr3viz)
library(precrec)

#for reproducability of the code
set.seed(123)

# ==============================================================================
# 2. DATA LOADING AND CLEANING
# ==============================================================================

#loading the data set
wdidata <- read.csv("WDICSV.csv")


#we list the needed variables from the data set
variables <- c(
  "ER.H2O.FWIN.ZS",
  "EG.USE.PCAP.KG.OE",
  "EG.USE.ELEC.KH.PC",
  "IT.NET.USER.ZS",
  "IT.NET.BBND.P2",
  "NY.GDP.PCAP.CD",
  "SP.POP.TOTL",
  "ER.H2O.FWAG.ZS",
  "IT.CEL.SETS.P2")


#filter the variables chosen from the indicator.code column
dataframe <- wdidata %>%
  filter(`Indicator.Code` %in% variables)



#extract year 2022, indicator code and country name
dataframe <- dataframe %>%
  select(Country.Name, Indicator.Code, X2022) %>%
  rename(
    Country = Country.Name,
    Value_in_2022 = X2022
  )

#convert from tall to wide format

wide_dataframe <- dataframe %>%
  pivot_wider(
    names_from = Indicator.Code,
    values_from = Value_in_2022
  )

#create clean data set by remove missing values
df_clean <- wide_dataframe %>% na.omit()

#rename the variables
df_clean <- df_clean %>%
  rename(
    Water = ER.H2O.FWIN.ZS,
    Energy = EG.USE.PCAP.KG.OE,
    Electricity = EG.USE.ELEC.KH.PC,
    Internet = IT.NET.USER.ZS,
    Broadband = IT.NET.BBND.P2,
    GDP = NY.GDP.PCAP.CD,
    Population = SP.POP.TOTL,
    Mobile_subscriptions = IT.CEL.SETS.P2,
    Agriculture = ER.H2O.FWAG.ZS)

# ==============================================================================
#                 3-  Exploratory Data Analysis "EDA"
# ==============================================================================

#Variable Description: creating histograms, one for each variable
# setting the layout
par(mfrow = c(3, 3))

# loop through each column
for(i in 2:ncol(df_clean)) {
  hist(df_clean[[i]],
       main = colnames(df_clean)[i],
       xlab = "",
       col = "#BAFFC9")
}

#Correlation Analysis: correlation matrix
numeric_data <- df_clean[, -1]
cor_matrix <- cor(numeric_data)

# to keep the lower triangle only
cor_matrix[upper.tri(cor_matrix)] <- NA

# convert to long format
cor_long <- melt(cor_matrix, na.rm = TRUE)

# plot
ggplot(cor_long, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  geom_text(aes(label = round(value, 2)), size = 3) +
  scale_fill_gradient2(low = "#E57373", mid = "#FFFFBA", high ="#5CCF8A" ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "", y = "", fill = "Correlation")

#summary statistics:
# remove Country column
numeric_data <- df_clean[, -1]

# create summary table
summary_table <- data.frame(
  Variable = colnames(numeric_data),
  Mean = sapply(numeric_data, mean),
  SD = sapply(numeric_data, sd),
  Min = sapply(numeric_data, min),
  Max = sapply(numeric_data, max)
)

# print it
print(summary_table)

# ==============================================================================
#           4- UNSUPERVISED LEARNING (1 out of 2): K-MEANS CLUSTERING
# ==============================================================================

#for k - means clustering we Keep only numeric variables

df_kmeans <- df_clean %>%
  select(-Country)


#Removing outliers using the IQR method, necessary for clustering
remove_out <- function(df) {
  df %>%
    filter(
      if_all(
        everything(),
        ~ . >= quantile(., 0.25) - 1.5 * IQR(.) &
          . <= quantile(., 0.75) + 1.5 * IQR(.)
      )
    )
}

df_kmeans_clean <- remove_out(df_kmeans)


#Standardize the data
df_scaled <- scale(df_kmeans_clean)


# conducting and graphing the Elbow method to determine the best no of clusters "k"
fviz_nbclust(
  df_scaled,
  kmeans,
  method = "wss",
  nstart = 20
) +
  labs(title = "Elbow Method for the Optimal K")


# Choose best K based on the graph, and run K-means

K <- 3  #change based on what the graph gives
km_out <- kmeans(df_scaled, centers = K, nstart = 20)


# summary of the clusters

km_out

# Profiling the clusters:

# Adding the  clusters to the data
df_with_clusters <- df_kmeans_clean
df_with_clusters$Cluster <- km_out$cluster


# Calculate cluster means
cluster_means <- aggregate(. ~ Cluster, data = df_with_clusters, mean)

# Add cluster size
cluster_size <- table(df_with_clusters$Cluster)
cluster_means$N <- as.vector(cluster_size)

print(cluster_means)

# Creating a table to show the number of countries in each cluster
#focusing on technology variables only

clusters_table <- data.frame(
  Cluster = cluster_means$Cluster,
  N = cluster_means$N,
  Energy = round(cluster_means$Energy, 1),
  Electricity = round(cluster_means$Electricity, 1),
  Internet = round(cluster_means$Internet, 1),
  Broadband = round(cluster_means$Broadband, 1),
  Mobile_subscriptions = round(cluster_means$Mobile_subscriptions,1),
  Water = round(cluster_means$Water, 1)
)

print(clusters_table)


# creating a box plot to visualize Water withdrawal by cluster
boxplot(Water ~ Cluster, data = df_with_clusters,
        main = "Water Withdrawal By Technology Cluster",
        xlab = "Cluster", ylab = "Water withdrawal (m³ per capita)",
       col = c("#E57373", "#FFFFBA", "#BAFFC9") )




# ==============================================================================
#                       5- Regression TASK
# ==============================================================================

# ==============================================================================
# Data Preparation:avoiding multicollenarity and log transformations
# ==============================================================================


# log transform the variables to use when needed
df_clean <- df_clean %>%
  mutate(
    log_gdp = log(GDP),
    log_water=log(Water),
    log_energy=log(Energy),
    log_electricity=log(Electricity),
    log_internet= log(Internet),
    log_population=log(Population),
    log_agriculture = log(Agriculture+1)
    
  )

# ==========================================================================
# Unsupervised learning (2 out of 2): Dimensionality Reduction 
# Principle Component Analysis "PCA"
# ==========================================================================


#preforming PCA
# Select only the technology-related variables - highly correlated ones
tech_vars <- df_clean[, c("Energy", "Electricity", 
                          "Internet", "Broadband","Mobile_subscriptions")]

#scaling the variables
techvars_scaled <- scale(tech_vars)

# Perform PCA
PCA <- prcomp(techvars_scaled)

# Look at the summary of the results
summary(PCA)
PCA$rotation

# since PCA 1 explains most of the variations (61.12)
#we will use its loadings to create tech index

df_clean$Tech_Index <-PCA$x[, 1]


#we now check the correlation of the tech index and the remaining variables
# Select your key variables
key_vars <- df_clean[, c("Tech_Index", "GDP", "Agriculture", "Population")]

# Calculate correlation matrix
cor_matrix <- cor(key_vars, use = "complete.obs")

# Print rounded to 3 decimals
print(round(cor_matrix, 3))
# ==========================================================================
#     Model 1: Linear regression with tech index + logged variables
# ==========================================================================

# define the model

model_1 <- lm(log_water ~ Tech_Index + Agriculture + log_population
                         , data=df_clean)

#summary of the results
summary(model_1)

#Plot of the residuals against the fitted values

plot(model_1$fitted.values, residuals(model_1),
     main = "Linear Regression",
     xlab = "Fitted Values", ylab = "Residuals",
     pch = 19, col ="#9AE6A8" )
abline(h = 0, col ="#E57373" , lwd = 2)

#do cross validation
train_control1 <- trainControl(method = "cv", number = 10)

# Define your model
task1 <- log_water ~ Tech_Index + Agriculture + log_population

model_cv1 <- train(task1, 
                  data = df_clean,
                  method = "lm",
                  trControl = train_control1)

# View results
print(model_cv1) #overall results
model_cv1$results #average RMSE, R squared, MAE
model_cv1$resample  # Results for each fold

#checking hetroskedasctity
bptest(model_1)

# the model with robust standard errors
coeftest(model_1, vcov = vcovHC(model_1, type = "HC3"))

# Compare with original
summary(model_1)$coefficients

# ==========================================================================
#                      Regularization methods
# ==========================================================================

# Define response variable for both ridge &lasso
y <- df_clean$log_water

# Define predictor matrix without the intercept
X <- model.matrix(
  log_water ~ Tech_Index + log_population +Agriculture,
  data = df_clean
)[, -1]


# ==========================================================================
#  Model 2- Ridge
# ==========================================================================


#define the model with cross validation
ridge_cv <- cv.glmnet(
  X, y,
  alpha = 0,        # 0 for Ridge
  nfolds = 10,      # no of folds in cv
  standardize = TRUE
)

ridge_cv$lambda.min     # the lambada that gives the lowest error
ridge_cv$lambda.1se     # higher error here, but simpler model

#RMSE
ridge_rmse <- sqrt(min(ridge_cv$cvm))
ridge_rmse

coef(ridge_cv, s = "lambda.min")

#Plot of the residuals against the fitted values 
ridge_pred <- predict(ridge_cv, s = "lambda.min", newx = X)
ridge_resid <- y - ridge_pred
plot(ridge_pred, ridge_resid,
     main = "Ridge Regression",
     xlab = "Fitted Values", ylab = "Residuals",
     pch = 19, col = "#8FB8E8")
abline(h = 0, col = "#E57373", lwd = 2)


# ==========================================================================
# Model 3 - Lasso
# ==========================================================================

#lasso regression

lasso_cv <- cv.glmnet(
  X, y,
  alpha = 1,        # 1 for Lasso
  nfolds = 10, # no of folds in cross  validation
  standardize = TRUE
)

lasso_cv$lambda.min # the lambada that gives the lowest error
lasso_cv$lambda.1se # higher error here, but simpler model

#RMSE
lasso_rmse <- sqrt(min(lasso_cv$cvm))
lasso_rmse

#view the coefficients with the min lambda
coef(lasso_cv, s = "lambda.min")


#Plot of the residuals against the fitted values 
lasso_pred <- predict(lasso_cv, s = "lambda.min", newx = X)
lasso_resid <- y - lasso_pred
plot(lasso_pred, lasso_resid,
     main = "Lasso Regression",
     xlab = "Fitted Values", ylab = "Residuals",
     pch = 19, col = "#E6E6A7")
abline(h = 0, col = "#E57373", lwd = 2)
# =============================================================================
#          Coefficient comparison: OLS vs Ridge vs Lasso
# =============================================================================

coef_compare <- data.frame(
  Predictor = c("Tech_Index", "log_population", "Agriculture"),
  
  OLS = coef(model_1)[c("Tech_Index", "log_population", "Agriculture")],
  
  Ridge = as.numeric(
    coef(ridge_cv, s = "lambda.min")[
      c("Tech_Index", "log_population", "Agriculture"), 1
    ]
  ),
  
  Lasso = as.numeric(
    coef(lasso_cv, s = "lambda.min")[
      c("Tech_Index", "log_population", "Agriculture"), 1
    ]
  )
)

# round for report presentation
coef_compare[, -1] <- round(coef_compare[, -1], 3)


# make them in a nice table
coef_compare %>%
  gt() %>%
  tab_header(
    title = "Coefficient Comparison Across Regression Models"
  ) %>%
  fmt_number(
    columns = c(OLS, Ridge, Lasso),
    decimals = 3
  ) %>%
  data_color(
    columns = c(OLS, Ridge, Lasso),
    method = "numeric",
    palette = c("#E57373", "#FFFFBA", "#BAFFC9")
  ) %>%
  cols_label(
    Predictor = "Predictor",
    OLS = "OLS",
    Ridge = "Ridge",
    Lasso = "Lasso"
  )

# ==============================================================================
#                Model 4 : RANDOM FOREST REGRESSION
# ==============================================================================

#since random forests are designed for non linear relationships, we added more variables
# and  didn't log them to see how well it can predict the water withdrawal


rf_formula <- Water ~ Tech_Index + GDP + Agriculture + Population

# 10-fold cross validation
train_control_rf <- trainControl(method = "cv", number = 10)

# train random forest
rf_model <- train(
  rf_formula,
  data = df_clean,
  method = "rf",
  trControl = train_control_rf,
  importance = TRUE
)

# view results
print(rf_model)

# fold-by-fold CV results
rf_model$resample

# average performance
rf_model$results


# look at importance scores
rf_importance <- randomForest(
  Water ~ Tech_Index + GDP + Agriculture + Population, 
  data = df_clean, 
  importance = TRUE
)

varImpPlot(rf_importance)



# ==============================================================================
#                             Model Comparisons- Regression
# ==============================================================================
# create a table comparing RMSE and R squared for all models
cv_table <- tibble(
  Model = c("OLS", "Ridge", "Lasso", "Random Forest"),
  
  CV_RMSE = c(
    model_cv1$results$RMSE,
    sqrt(min(ridge_cv$cvm)),
    sqrt(min(lasso_cv$cvm)),
    min(rf_model$results$RMSE)
  ),
  
  CV_R2 = c(
    model_cv1$results$Rsquared,
    1 - min(ridge_cv$cvm)/var(y),
    1 - min(lasso_cv$cvm)/var(y),
    max(rf_model$results$Rsquared))
)

cv_table %>%
  gt() %>%
  tab_header(
    title = "Cross-Validated Model Performance Comparison",
    subtitle = "10-Fold CV metrics for all predictive models"
  ) %>%
  fmt_number(
    columns = c(CV_RMSE, CV_R2),
    decimals = 3
  ) %>%
  data_color(
    columns = c(CV_RMSE),
    method = "numeric",
    palette = c("#BAFFC9", "#FFFFBA", "#FFB3BA")
  ) %>%
  data_color(
    columns = c(CV_R2),
    method = "numeric",
    palette = c("#FFB3BA", "#FFFFBA", "#BAFFC9")
  ) %>%
  cols_label(
    CV_RMSE = "CV RMSE",
    CV_R2 = "CV R²"
  )

# ==============================================================================
# back transforming Linear Models using bias-corrected exponential transformation
# ==============================================================================

#linear regression back transformation
train_control1 <- trainControl(
  method = "cv", 
  number = 10,
  savePredictions = "all"
)

model_cv1 <- train(
  log_water ~ Tech_Index + Agriculture + log_population, 
  data = df_clean,
  method = "lm",
  trControl = train_control1
)
ols_cv <- model_cv1$pred
ols_cv$pred <- as.numeric(ols_cv$pred)
ols_cv$obs  <- as.numeric(as.character(ols_cv$obs))
sigma2_ols <- var(residuals(model_1))
ols_cv$pred_water <- exp(ols_cv$pred + sigma2_ols / 2)
ols_cv$obs_water  <- exp(ols_cv$obs)
rmse_ols_cv <- rmse(ols_cv$obs_water, ols_cv$pred_water)
mae_ols_cv  <- mae(ols_cv$obs_water, ols_cv$pred_water)
r2_ols_cv   <- cor(ols_cv$obs_water, ols_cv$pred_water)^2


# Lasso back-transformation
y_numeric <- as.numeric(y)
lasso_cv_pred <- predict(lasso_cv, newx = X, s = "lambda.min")
lasso_cv_pred <- as.numeric(lasso_cv_pred)
lasso_pred_water <- exp(lasso_cv_pred + sigma2_ols / 2)
lasso_obs_water  <- exp(y_numeric)
rmse_lasso_cv <- rmse(lasso_obs_water, lasso_pred_water)
mae_lasso_cv  <- mae(lasso_obs_water, lasso_pred_water)
r2_lasso_cv   <- cor(lasso_obs_water, lasso_pred_water)^2

# Lasso back transformation
ridge_cv_pred <- predict(ridge_cv, newx = X, s = "lambda.min")
ridge_cv_pred <- as.numeric(ridge_cv_pred)
ridge_pred_water <- exp(ridge_cv_pred + sigma2_ols / 2)
ridge_obs_water  <- exp(y_numeric)
rmse_ridge_cv <- rmse(ridge_obs_water, ridge_pred_water)
mae_ridge_cv  <- mae(ridge_obs_water, ridge_pred_water)
r2_ridge_cv   <- cor(ridge_obs_water, ridge_pred_water)^2


# Random Forest CV metrics (already in Water scale)
rf_rmse_cv <- min(rf_model$results$RMSE)
rf_mae_cv  <- rf_model$results$MAE[rf_model$results$mtry == rf_model$bestTune$mtry]
rf_r2_cv   <- max(rf_model$results$Rsquared)

# Combine the metrics in one table

comparison_cv_water <- tibble(
  Model = c("OLS", "Ridge", "Lasso", "Random Forest"),
  CV_RMSE = c(rmse_ols_cv, rmse_ridge_cv, rmse_lasso_cv, rf_rmse_cv),
  CV_R2   = c(r2_ols_cv, r2_ridge_cv, r2_lasso_cv, rf_r2_cv)
)

comparison_cv_water %>%
  gt() %>%
  tab_header(
    title = "Cross-Validated Model Comparison on Water Scale",
    subtitle = "Bias-corrected back-transformation for log-based models"
  ) %>%
  fmt_number(
    columns = c(CV_RMSE, CV_R2),
    decimals = 3
  ) %>%
  data_color(
    columns = CV_RMSE,
    method = "numeric",
    palette = c("#BAFFC9", "#FFFFBA", "#FFB3BA")
  ) %>%
  data_color(
    columns = CV_R2,
    method = "numeric",
    palette = c("#FFB3BA", "#FFFFBA", "#BAFFC9")
  )
# ==============================================================================
#                            6-    CLASSIFICATION TASK
# ==============================================================================

# ============================================
# Creating binary variable as a target
# ============================================

# 1 = high withdrawal
# 0 = low withdrawal

df_clean$high_withdrawal <- factor(
  ifelse(df_clean$Water >= 25, 1, 0)
)

# ==============================================================================
#                              Logistic Regression
# ==============================================================================
# To see if the predictors can naturally have separation

# Create withdrawal group labels
df_clean$withdrawal_group <- ifelse(
  df_clean$Water >= 25,
  "High Withdrawal",
  "Low Withdrawal"
)

# Scatter plot: log GDP vs Agriculture
ggplot(df_clean, aes(x = log_gdp, y = Agriculture, color = withdrawal_group)) +
  geom_point(size = 3, alpha = 0.8) +
  labs(
    title = " Agriculture vs GDP per Capita by Classification Group",
    x = "Log GDP per Capita",
    y = "Agriculture water share",
    color = "Group"
  ) +
  theme_minimal()


# specify the independent variables

df_class <- df_clean %>%
  select(
    high_withdrawal,
    Tech_Index,
    log_gdp,
    Agriculture,
    log_population
  ) %>%
  na.omit()


# Logistic Regression task with cross validation
task <- as_task_classif(
  df_class,
  target = "high_withdrawal",
  positive = "1"
)


logistic <- lrn(
  "classif.log_reg",
  predict_type = "prob"
)


# preforming 10-fold cross validation using benchmarking
design <- benchmark_grid(
  tasks = task,
  learners = logistic,
  resamplings = rsmps("cv", folds = 10)
)

bmr <- benchmark(design)


# viewing all the metrics

all_measures <- list(
  msr("classif.auc"),        # ROC AUC
  msr("classif.precision"),  # Precision
  msr("classif.recall"),     # Recall
  msr("classif.fbeta", beta = 1), # F1
  msr("classif.acc"),        # Accuracy
  msr("classif.ce")          # Classification error
)

logistic_results <- bmr$aggregate(all_measures)

print(logistic_results)

# ROC Curve
autoplot(bmr, type = "roc")

# Precision-Recall Curve
autoplot(bmr, type = "prc")
# ==============================================================================
#                              Random Forests
# ==============================================================================

# Create binary target, since we are using the caret library here
df_clean$high_withdrawal <- factor(
  ifelse(df_clean$Water >= 25, "high", "low"),
  levels = c("low", "high")
)

# define the independent variables
df_rf_class <- df_clean %>%
  select(
    high_withdrawal,
    Tech_Index,
    log_gdp,
    Agriculture,
    log_population
  ) %>%
  na.omit()


# define the Random forest formula
rf_formula_class <- high_withdrawal ~
  Tech_Index + log_gdp + Agriculture + log_population

# preforming 10-fold cross validation
train_control_rf <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  savePredictions = "final",
  summaryFunction = twoClassSummary
)


# Training the model
rf_class_model <- train(
  rf_formula_class,
  data = df_rf_class,
  method = "rf",
  metric = "ROC",
  trControl = train_control_rf,
  importance = TRUE
)

#view results
print(rf_class_model)

# view the fold-by-fold cross validation results
print(rf_class_model$resample)

# view the average of the cross validation results
print(rf_class_model$results)


# RANDOM FOREST METRICS

rf_preds <- rf_class_model$pred

# Use best tuning parameter only
best_mtry <- rf_class_model$bestTune$mtry

rf_preds_best <- rf_preds %>%
  filter(mtry == best_mtry)

# Confusion matrix
cm <- confusionMatrix(
  rf_preds_best$pred,
  rf_preds_best$obs,
  positive = "high"
)

# Extracting metrics
rf_precision <- as.numeric(cm$byClass["Pos Pred Value"])
rf_recall <- as.numeric(cm$byClass["Sensitivity"])
rf_f1 <- 2 * (rf_precision * rf_recall) / (rf_precision + rf_recall)
rf_roc <- as.numeric(rf_class_model$results$ROC[
  rf_class_model$results$mtry == best_mtry
])

rf_metrics <- data.frame(
  ROC = round(rf_roc, 3),
  Precision = round(rf_precision, 3),
  Recall = round(rf_recall, 3),
  F1 = round(rf_f1, 3)
)

#view the preformance measures
print(rf_metrics)

#ROC plot
rf_roc_curve <- roc(
  response = rf_preds_best$obs,
  predictor = rf_preds_best$high,
  levels = c("low", "high")
)

plot(
  rf_roc_curve,
  main = "Random Forest ROC Curve"
)

# Precision -Recall curve

pr <- pr.curve(
  scores.class0 = rf_preds_best$high[rf_preds_best$obs == "high"],
  scores.class1 = rf_preds_best$high[rf_preds_best$obs == "low"],
  curve = TRUE
)

plot(pr, 
     main = "Random Forest Precision-Recall Curve",
     col = "black", 
     lwd = 2,       
     auc.main = FALSE)  

# ==============================================================================
#                            XGBOOST CLASSIFICATION
# ==============================================================================

# defining the task
task_xgb <- as_task_classif(
  df_rf_class,
  target = "high_withdrawal",
  positive = "high"
)

# XGBoost learner

xgb_learner <- lrn(
  "classif.xgboost",
  predict_type = "prob"
)

#setting the parameters
xgb_learner$param_set$values <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  nrounds = 100,
  eta = 0.05,
  max_depth = 4,
  subsample = 0.8,
  colsample_bytree = 0.8,
  verbose = 0
)

#preforming 10 fold cross validation using benchmarking
xgb_design <- benchmark_grid(
  tasks = task_xgb,
  learners = xgb_learner,
  resamplings = rsmps("cv", folds = 10)
)

xgb_bmr <- benchmark(xgb_design)

#preformance metrices
xgb_measures <- list(
  msr("classif.auc"),
  msr("classif.precision"),
  msr("classif.recall"),
  msr("classif.fbeta", beta = 1),
  msr("classif.acc"),
  msr("classif.ce")
)

#ROC curve
autoplot(
  xgb_bmr,
  type = "roc"
) +
  ggtitle("XGBoost ROC Curve")

xgb_results <- xgb_bmr$aggregate(xgb_measures)

print(xgb_results)

# Precision-Recall curve
autoplot(
  xgb_bmr,
  type = "prc"
) +
  ggtitle("XGBoost Precision-Recall Curve")

#============================================================================
#                             Model comparisons- Classification
#============================================================================

# Creating a table for metric comparison

# Logistic Regression
logistic_df <- logistic_results %>%
  as.data.frame() %>%
  mutate(Model = "Logistic Regression") %>%
  select(Model, classif.auc, classif.precision, classif.recall, classif.fbeta) %>%
  rename(
    ROC = classif.auc,
    Precision = classif.precision,
    Recall = classif.recall,
    F1 = classif.fbeta
  ) %>%
  mutate(across(ROC:F1, ~round(., 3)))

# Random Forest
rf_df <- rf_metrics %>%
  mutate(Model = "Random Forest") %>%
  select(Model, ROC, Precision, Recall, F1)

# XGBoost
xgb_df <- xgb_results %>%
  as.data.frame() %>%
  mutate(Model = "XGBoost") %>%
  select(Model, classif.auc, classif.precision, classif.recall, classif.fbeta) %>%
  rename(
    ROC = classif.auc,
    Precision = classif.precision,
    Recall = classif.recall,
    F1 = classif.fbeta
  ) %>%
  mutate(across(ROC:F1, ~round(., 3)))

# Combine all models
all_models_df <- bind_rows(logistic_df, rf_df, xgb_df)

gt(all_models_df) %>%
  tab_header(
    title = "Classification Model Performance Metrics"
  ) %>%
  fmt_number(
    columns = ROC:F1,
    decimals = 3
  ) %>%
  cols_label(
    F1 = "F1 Score"
  ) %>%
  tab_options(
    table.font.names = "Arial",
    heading.title.font.size = 16,
    column_labels.font.weight = "bold"
  )

# Create a bar chart comparing metrics
# Your data
metrics_data <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "XGBoost"),
  ROC_AUC = c(0.938, 0.951, 0.958),
  Precision = c(0.810, 0.784, 0.757),
  Recall = c(0.790, 0.763, 0.780),
  F1 = c(0.774, 0.773, 0.746)
)

# Reshape for plotting
metrics_long <- pivot_longer(metrics_data, 
                             cols = -Model, 
                             names_to = "Metric", 
                             values_to = "Score")

ggplot(metrics_long, aes(x = Model, y = Score, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge(0.9)) +
  scale_fill_manual(values = c("ROC_AUC" = "#5CCF8A", 
                               "Precision" = "#E57373", 
                               "Recall" = "#FFFFBA",
                               "F1" = "#87CEEB")) +
  labs(title = "Model Performance Comparison",
       x = "", y = "Score",
       fill = "Metric") +
  theme_minimal() +
  theme(legend.position = "bottom") +
  coord_cartesian(ylim = c(0.7, 1.0))