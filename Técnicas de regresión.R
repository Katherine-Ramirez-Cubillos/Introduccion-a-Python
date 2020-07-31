#Taller 2 t閏nicas de regresi髇 
#Sebasti醤 Estepa
#Katherine Ram韗ez

rm(list=ls())

library(dplyr)
library(psych)
library(GGally)
library(ggplot2)
library(gridExtra)
library(lmtest)
library(corrplot)
library(car)
library(ISLR)
require(knitr)
library(AER)
library(stargazer)
library(scales)
library(MASS)
library(mvtnorm)
library(rsample)
library(e1071)
library(tree) 
library(rpart) 
library(rpart.plot)
library(C50) 
library(ggpubr)
library(randomForest)
library(tidyverse)
library(ggpubr)
library(randomForest)
library(tidyverse)
library(randomForest)
library(gbm)
library(caret)
library(kernlab)
library(LiblineaR)

options(scipen=999)

getwd()
setwd("C:/Users/Acer/Desktop/CURSOS D蒀IMO SEMESTRE/Machine Learning/C骴igos/Taller")

# Loading Database
datos <- read.csv('NBA.csv', sep = ',')
datos <- subset(datos, Player != 'Kay Felder')

# Setting Dummies
countries = data.frame(unique(datos[, 'NBA_Country']))
colnames(countries) <- 'Test'
datos <- datos %>% mutate(AmericaS = case_when(NBA_Country == 'Brazil' ~ 1, NBA_Country == 'Argentina' ~ 1, TRUE ~ 0))
datos <- datos %>% mutate(Europe = case_when(NBA_Country == 'Georgia' ~ 1, NBA_Country == 'Slovenia' ~ 1, NBA_Country == 'Greece' ~ 1, NBA_Country == 'Bosnia & Herz...' ~ 1, NBA_Country == 'Sweden' ~ 1, NBA_Country == 'Ukraine' ~ 1, NBA_Country == 'Austria' ~ 1, NBA_Country == 'Finland' ~ 1, NBA_Country == 'Latvia' ~ 1, NBA_Country == 'Bosnia' ~ 1, NBA_Country == 'Lithuania' ~ 1, NBA_Country == 'Croatia' ~ 1, NBA_Country == 'Italy' ~ 1, NBA_Country == 'Poland' ~ 1, NBA_Country == 'Montenegro' ~ 1, NBA_Country == 'Serbia' ~ 1, NBA_Country == 'United Kingdo...' ~ 1, NBA_Country == 'Germany' ~ 1, NBA_Country == 'Spain' ~ 1, NBA_Country == 'Switzerland' ~ 1, NBA_Country == 'France' ~ 1, NBA_Country == 'Czech Republic' ~ 1, TRUE ~ 0))
datos <- datos %>% mutate(Oceania = case_when(NBA_Country == 'New Zealand' ~ 1, NBA_Country == 'Australia' ~ 1, TRUE ~ 0))
datos <- datos %>% mutate(Asia = case_when(NBA_Country == 'China' ~ 1, NBA_Country == 'Israel' ~ 1, NBA_Country == 'Turkey' ~ 1, NBA_Country == 'Russia' ~ 1, TRUE ~ 0))
datos <- datos %>% mutate(Africa = case_when(NBA_Country == 'Egypt' ~ 1, NBA_Country == 'Democratic Re...' ~ 1, NBA_Country == 'Mali' ~ 1, NBA_Country == 'Senegal' ~ 1, NBA_Country == 'South Sudan' ~ 1, NBA_Country == 'Cameroon' ~ 1, NBA_Country == 'Democratic Re_' ~ 1, NBA_Country == 'Tunisia' ~ 1, TRUE ~ 0))

# Renaming Rows
rownames(datos) <- datos$Player

# Removing Strings
datos <- subset(datos, select = -c(Player, NBA_Country, Tm))
datos <- datos %>% na.omit()

# Scaling Data
datos <- data.frame(scale(datos))


# Removing Atypical Data
datos <- datos[setdiff(rownames(datos), c('Paul Millsap', 'Kyrie Irving', 'Karl-Anthony Towns', 'Gordon Hayward', 'Ben Simmons', 'Aron Baynes',
                                          'Steven Adams', 'Manu Ginobili', 'Dante Exum', 'Chandler Parsons', 'Blake Griffin',
                                          'Naz Mitrou-Long', 'Mike Conley', 'James Harden', 'Dirk Nowitzki',
                                          'Vince Hunter', 'Stephen Curry', 'Nikola Jokic', 'Mindaugas Kuzminskas',
                                          'Nicolas Batum', 'Luis Montero', 'Jeremy Lin', 'David West', 'David Stockton', 'Danilo Gallinari')),]

# Splitting Data
dt_split <- initial_split(datos, prop = .75)
dt_train <- training(dt_split)
dt_test  <- testing(dt_split)

# No multicolinealidad (Eliminaci髇 de variables correlacionadas)
corrplot(cor(dplyr::select(dt_train, Salary, NBA_DraftNumber, Age, MP,
                           PER, X3PAr, FTr, ORB., DRB., AST., STL.,
                           BLK., TOV., USG., OWS, DBPM,
                           AmericaS, Oceania, Europe, Africa, Asia)),
         method = "number", tl.col = "black")

##-----Regresi髇 Lineal-----###

# Model with all non-correlated variables
modelo <- lm(formula = Salary ~ NBA_DraftNumber + Age + MP +
               PER + X3PAr + FTr + ORB. + DRB. + AST. + STL. +
               BLK. + TOV. + USG. + OWS + DBPM +
               AmericaS + Asia + Oceania + Europe + Africa, data = dt_train)
summary(modelo)

# Selecci髇 del mejor modelo dado el criterio AIC
step(object = modelo, direction = "both", trace = 1)

modelo <- lm(formula = Salary ~ NBA_DraftNumber + Age + MP + PER + DRB. + 
               TOV. + USG. + OWS + DBPM + Africa, data = dt_train)
summary(modelo)

modelo <- lm(formula = Salary ~ 0 + NBA_DraftNumber + Age + MP + PER + DRB. + 
               TOV. + USG. + OWS + DBPM + Africa, data = dt_train)
summary(modelo)
plot(modelo)

# Normalidad
qqnorm(modelo$residuals)
qqline(modelo$residuals)

shapiro.test(modelo$residuals) # H0: Normalidad

#Homocedasticidad
ggplot(data = dt_train, aes(modelo$fitted.values, modelo$residuals)) +
  geom_point() +
  geom_smooth(color = "firebrick", se = FALSE) +
  geom_hline(yintercept = 0) +
  theme_bw()

# Breusch-Pagan test
bptest(modelo) # H0: Homoced醩tico

# No multicolinealidad
corrplot(cor(dplyr::select(dt_train, Salary, NBA_DraftNumber, Age, MP,
                           ORB., STL., USG., OWS, DBPM)),
         method = "number", tl.col = "black")


# Analisis de inflacion de la varianza

vif(modelo) # Por debajo de 4, no hay problemas de colinealidad

# Autocorrelaci髇
dwt(modelo, alternative = "two.sided") # Durbin-Watson -> H0 : rho = 0, No hay relacion entre er0 y er1, no hay problemas de autocorrelaci髇  

# Grafico de influencias

influencePlot(modelo) # Valores at韕icos -> Estandarizar variables, restar media y dividir por desviaci髇 estandar de todos los valores

# Predicci髇
predicciones_lm <- predict(object = modelo, newdata = dt_test)
test_lm <- mean((predicciones_lm - dt_test[, 'Salary'])^2)
paste('Error de test (mse) del modelo de regresi髇 lineal es:', (round(test_lm, 2)))

##---- M醧uina soprtada en vectores ----###
modelo_svm <- svm(formula = Salary ~ 0 + NBA_DraftNumber + Age + MP + PER + DRB. + 
                    TOV. + USG. + OWS + DBPM + Africa, data = dt_train, 
                  type = "eps-regression")
modelo_svm

predicciones_svm <- predict(object = modelo_svm, newdata = dt_test)
test_svm <- mean((predicciones_svm - dt_test[, 'Salary'])^2)
paste('Error de test (mse) del modelo de SVM es:', (round(test_svm, 2)))


### ----- 羠bol de decisi髇 ------ ###
arbol_regresion <- tree(formula = Salary ~ 0 + NBA_DraftNumber + Age + MP + PER + DRB. + 
                          TOV. + USG. + OWS + DBPM + Africa, data = dt_train, 
                        split = "deviance")
plot(x = arbol_regresion, type = "proportional")
text(x = arbol_regresion, splits = TRUE, pretty = 0,
     cex = 0.8, col = "firebrick")

set.seed(3)
cv_arbol <- cv.tree(arbol_regresion, K = 10)
cv_arbol


# C醠culo del Hiperpar醡etro
resultados_cv <- data.frame(n_nodos = cv_arbol$size,
                            deviance = cv_arbol$dev,
                            alpha = cv_arbol$k)

p1 <- ggplot(data = resultados_cv, aes(x = n_nodos, y = deviance)) +
  geom_line() + 
  geom_point() +
  labs(title = "Error vs tama帽o del 谩rbol") + theme_bw() 


p2 <- ggplot(data = resultados_cv, aes(x = alpha, y = deviance)) +
  geom_line() + 
  geom_point() +
  labs(title = "Error vs hiperpar谩metro alpha") + theme_bw() 

ggarrange(p1, p2)


arbol_pruning <- prune.tree(tree = arbol_regresion, best = 13)
plot(x = arbol_pruning, type = "proportional")
text(x = arbol_pruning, splits = TRUE, pretty = 0,
     cex = 0.8, col = "firebrick")

predicciones_tree <- predict(arbol_pruning, newdata = dt_test)
test_tree     <- mean((predicciones_tree - dt_test[, "Salary"])^2)
paste("Error de test (mse) del 谩rbol de regresi贸n tras podado:", (round(test_tree,2)))



###----------- Bagging------------------

modelo_bagging <- randomForest(formula = Salary ~ 0 + NBA_DraftNumber + Age + MP + PER + DRB. + 
                                 TOV. + USG. + OWS + DBPM + Africa, data = dt_train, mtry = 13)




modelo_bagging

predicciones_bagging <- predict(object = modelo_bagging, newdata = dt_test)
test_bagging     <- mean((predicciones - dt_test[, 'Salary'])^2)
paste("Error de test (mse) del modelo obtenido por bagging es:",
      (round(test_mse,2)))

# Importancia
modelo_bagging_1 <- randomForest(formula = Salary ~ 0 + NBA_DraftNumber + Age + MP + PER + DRB. + 
                                   TOV. + USG. + OWS + DBPM + Africa, data = dt_train,
                                 mtry = 13, ntree = 500,
                                 importance = TRUE)

importancia_pred <- as.data.frame(importance(modelo_bagging_1,
                                             scale = TRUE))
importancia_pred <- rownames_to_column(importancia_pred,
                                       var = "variable")

p1 <- ggplot(data = importancia_pred, 
             aes(x = reorder(variable, `%IncMSE`),
                 y = `%IncMSE`,fill = `%IncMSE`)) +
  labs(x = "variable", title = "Reducci贸n de MSE") +
  geom_col() +
  coord_flip() +
  theme_bw() +
  theme(legend.position = "bottom")

p2 <- ggplot(data = importancia_pred, 
             aes(x = reorder(variable, IncNodePurity),
                 y = IncNodePurity,
                 fill = IncNodePurity)) +
  labs(x = "variable", title = "Reducci贸n de pureza") +
  geom_col() +
  coord_flip() +
  theme_bw() +
  theme(legend.position = "bottom")
ggarrange(p1, p2)



###----- Random Forest ------###

tuning_rf_nodesize <- function(df, y, size = NULL, ntree = 500){
  if (is.null(size)){
    size <- seq(from = 1, to = nrow(df), by = 5)
  }
  oob_mse <- rep(NA, length(size))
  for (i in seq_along(size)) {
    set.seed(123)
    f <- formula(paste(y,"~ ."))
    modelo_rf <- randomForest(formula = f, data = df, mtry = 5,
                              ntree = ntree, nodesize = i)
    
    oob_mse[i] <- tail(modelo_rf$mse, n = 1)
  }
  results <- data_frame(size, oob_mse)
  return(results)
}

hiperparametro_nodesize <-  tuning_rf_nodesize(df = dt_test, y = "Salary",
                                               size = c(1:20))

hiperparametro_nodesize %>% arrange(oob_mse)

ggplot(data = hiperparametro_nodesize, aes(x = size, y = oob_mse)) +
  scale_x_continuous(breaks = hiperparametro_nodesize$size) +
  geom_line() +
  geom_point() +
  geom_point(data = hiperparametro_nodesize %>% arrange(oob_mse) %>% head(1),
             color = "red") +
  labs(title = "Evoluci贸n del out-of-bag-error vs nodesize",
       x = "n潞 observaciones en nodos terminales") +
  theme_bw()

modelo_randomforest <- randomForest(formula = Salary ~ 0 + NBA_DraftNumber + Age + MP + PER + DRB. + 
                                      TOV. + USG. + OWS + DBPM + Africa, data = dt_train, 
                                    mtry = 5 , 
                                    ntree = 500, nodesize = 2,
                                    importance = TRUE)

oob_mse <- data.frame(oob_mse = modelo_randomforest$mse,
                      arboles = seq_along(modelo_randomforest$mse))

ggplot(data = oob_mse, aes(x = arboles, y = oob_mse )) +
  geom_line() +
  labs(title = "Evoluci贸n del out-of-bag-error vs n煤mero 谩rboles",
       x = "n潞 谩rboles") +
  theme_bw()

# Ajuste final

predicciones_random <- predict(object = modelo_randomforest,
                               newdata = dt_test)

test_random <- mean((predicciones_random - dt_test[, "Salary"])^2)

paste("Error de test (mse) del modelo:", round(test_random, 2))


###--- Boosting ----###

cv_error  <- vector("numeric")
n_arboles <- vector("numeric")
shrinkage <- vector("numeric")

# Learning rate (lambda/shrinkage)

for (i in c(0.001, 0.01, 0.1)) {
  set.seed(123)
  arbol_boosting <- gbm(formula = Salary ~ 0 + NBA_DraftNumber + Age + MP + PER + DRB. + 
                          TOV. + USG. + OWS + DBPM + Africa, data = dt_train,
                        distribution = "gaussian",
                        n.trees = 20000,
                        interaction.depth = 1,
                        shrinkage = i,
                        n.minobsinnode = 10,
                        bag.fraction = 0.5,
                        cv.folds = 5)
  cv_error  <- c(cv_error, arbol_boosting$cv.error)
  n_arboles <- c(n_arboles, seq_along(arbol_boosting$cv.error))
  shrinkage <- c(shrinkage, rep(i, length(arbol_boosting$cv.error)))
}
error <- data.frame(cv_error, n_arboles, shrinkage)

ggplot(data = error, aes(x = n_arboles, y = cv_error,
                         color = as.factor(shrinkage))) +
  geom_smooth() +
  labs(title = "Evoluci贸n del cv-error", color = "shrinkage") + 
  theme_bw() +
  theme(legend.position = "bottom")

## Complejidad de los arboles (N煤mero de divisiones)

cv_error  <- vector("numeric")
n_arboles <- vector("numeric")
interaction.depth <- vector("numeric")
for (i in c(1, 3, 5, 10)) {
  set.seed(123)
  arbol_boosting <- gbm(formula = Salary ~ 0 + NBA_DraftNumber + Age + MP + PER + DRB. + 
                          TOV. + USG. + OWS + DBPM + Africa, data = dt_train,
                        distribution = "gaussian",
                        n.trees = 5000,
                        interaction.depth = i,
                        shrinkage = 0.01,
                        n.minobsinnode = 10,
                        bag.fraction = 0.5,
                        cv.folds = 5)
  cv_error  <- c(cv_error, arbol_boosting$cv.error)
  n_arboles <- c(n_arboles, seq_along(arbol_boosting$cv.error))
  interaction.depth <- c(interaction.depth,
                         rep(i, length(arbol_boosting$cv.error)))
}
error <- data.frame(cv_error, n_arboles, interaction.depth)

ggplot(data = error, aes(x = n_arboles, y = cv_error,
                         color = as.factor(interaction.depth))) +
  geom_smooth() +
  labs(title = "Evoluci贸n del cv-error", color = "interaction.depth") + 
  theme_bw() +
  theme(legend.position = "bottom")

## Minimo numero de observaciones por nodo

cv_error  <- vector("numeric")
n_arboles <- vector("numeric")
n.minobsinnode <- vector("numeric")
for (i in c(1, 5, 10, 20)) {
  arbol_boosting <- gbm(formula = Salary ~ 0 + NBA_DraftNumber + Age + MP + PER + DRB. + 
                          TOV. + USG. + OWS + DBPM + Africa, data = dt_train,
                        distribution = "gaussian",
                        n.trees = 5000,
                        interaction.depth = 5,
                        shrinkage = 0.01,
                        n.minobsinnode = i,
                        bag.fraction = 0.5,
                        cv.folds = 5)
  cv_error  <- c(cv_error, arbol_boosting$cv.error)
  n_arboles <- c(n_arboles, seq_along(arbol_boosting$cv.error))
  n.minobsinnode <- c(n.minobsinnode,
                      rep(i, length(arbol_boosting$cv.error)))
}
error <- data.frame(cv_error, n_arboles, n.minobsinnode)

ggplot(data = error, aes(x = n_arboles, y = cv_error,
                         color = as.factor(n.minobsinnode))) +
  geom_smooth() +
  labs(title = "Evoluci贸n del cv-error", color = "n.minobsinnode") + 
  theme_bw() +
  theme(legend.position = "bottom")

# Numero de arboles

set.seed(123)
arbol_boosting <- gbm(formula = Salary ~ 0 + NBA_DraftNumber + Age + MP + PER + DRB. + 
                        TOV. + USG. + OWS + DBPM + Africa, data = dt_train,
                      distribution = "gaussian",
                      n.trees = 10000,
                      interaction.depth = 5,
                      shrinkage = 0.01,
                      n.minobsinnode = 1,
                      bag.fraction = 0.5,
                      cv.folds = 5)
error <- data.frame(cv_error = arbol_boosting$cv.error,
                    n_arboles = seq_along(arbol_boosting$cv.error))
ggplot(data = error, aes(x = n_arboles, y = cv_error)) +
  geom_line(color = "black") +
  geom_point(data = error[which.min(error$cv_error),], color = "red") +
  labs(title = "Evoluci贸n del cv-error") + 
  theme_bw() 

error[which.min(error$cv_error),]

# Se reajusta el modelo final con los hiperpar谩metro 贸ptimos
set.seed(123)
arbol_boosting <- gbm(formula = Salary ~ 0 + NBA_DraftNumber + Age + MP + PER + DRB. + 
                        TOV. + USG. + OWS + DBPM + Africa, data = dt_train,
                      distribution = "gaussian",
                      n.trees = 2791,
                      interaction.depth = 5,
                      shrinkage = 0.01,
                      n.minobsinnode = 1,
                      bag.fraction = 0.5)
# Hiperparametros optimos con caret

set.seed(123)
validacion <- trainControl(## 10-fold CV
  method = "cv",
  number = 10)

tuning_grid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                            n.trees = c(100, 1000, 2000, 3000), 
                            shrinkage = c(0.1, 0.01, 0.001),
                            n.minobsinnode = c(1, 10, 20))

set.seed(123)
mejor_modelo <- train(formula = Salary ~ 0 + NBA_DraftNumber + Age + MP + PER + DRB. + 
                        TOV. + USG. + OWS + DBPM + Africa, data = dt_train, 
                      method = "gbm", 
                      trControl = validacion, 
                      verbose = FALSE, 
                      tuneGrid = tuning_grid)

# Se muestran los hiperpar谩metros del mejor modelo 
mejor_modelo$bestTune


arbol_boosting <- gbm(formula = Salary ~ 0 + NBA_DraftNumber + Age + MP + PER + DRB. + 
                        TOV. + USG. + OWS + DBPM + Africa, data = dt_train,
                     distribution = "gaussian",
                     n.trees = 3000,
                     interaction.depth = 5,
                     shrinkage = 0.01,
                     n.minobsinnode = 1,
                     bag.fraction = 0.5)

## Importancia de cada variable

importancia_pred <- summary(arbol_boosting, plotit = FALSE)

ggplot(data = importancia_pred,
       aes(x = reorder(var, rel.inf), y = rel.inf,fill = rel.inf)) +
  labs(x = "variable", title = "Reducci贸n de MSE") +
  geom_col() +
  coord_flip() +
  theme_bw() +
  theme(legend.position = "bottom")

par(mfrow = c(1,2))

## Influencia de algunos preductores sobre la variable y
plot(arbol_boosting, i.var = "MP", col = "blue")
plot(arbol_boosting, i.var = "Age", col = "firebrick")


## Predicciones
predicciones_boosting <- predict(object = arbol_boosting, 
                        newdata = dt_test,
                        n.trees = 337)

test_boosting <- mean((predicciones_boosting - dt_test[, 'Salary'])^2)
paste("Error de test (mse) del modelo boosing:", round(test_boosting, 2))






