# --------------------------
# PUNTO 1: LIBRERÍAS Y PARÁMETROS
# --------------------------
library(magrittr)
library(ggplot2)
library(shiny)
library(nnet)

# --------------------------
# PUNTO 2: GENERACIÓN DE DATOS EN ESPIRAL
# --------------------------
set.seed(308)  # Semilla reproducible

N <- 200
D <- 2
K <- 4
X <- data.frame()
Y_raw <- data.frame()

for (j in 1:K) {
  r <- seq(0.05, 1, length.out = N)
  t <- seq((j - 1) * 4.7, j * 4.7, length.out = N) + rnorm(N, sd = 0.3)
  Xtemp <- data.frame(x1 = r * sin(t), x2 = r * cos(t))
  Ytemp <- data.frame(label = rep(j, N))
  X <- rbind(X, Xtemp)
  Y_raw <- rbind(Y_raw, Ytemp)
}

# Combina X e Y
data <- cbind(X, Y_raw)

# --------------------------
# PUNTO 2: CODIFICACIÓN ONE-HOT
# --------------------------
Y <- data.frame(matrix(0, nrow = nrow(data), ncol = K))
for (i in 1:nrow(data)) {
  Y[i, data$label[i]] <- 1
}
colnames(Y) <- paste("class", 1:K, sep = "")
X <- data.frame(x1 = data$x1, x2 = data$x2)

# --------------------------
# PLOT DE LOS DATOS GENERADOS
# --------------------------
x1_min <- min(X$x1) - 0.2
x1_max <- max(X$x1) + 0.2
x2_min <- min(X$x2) - 0.2
x2_max <- max(X$x2) + 0.2

puntos <- ggplot() +  
  geom_point(data = data, aes(x = x1, y = x2, color = as.character(label)), size = 2) +
  theme_bw(base_size = 15) +
  xlim(x1_min, x1_max) +
  ylim(x2_min, x2_max) +
  ggtitle('Visualización de Datos en Espiral') +
  coord_fixed(ratio = 1) +
  theme(
    axis.ticks = element_blank(), 
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    axis.text = element_blank(), 
    axis.title = element_blank(), 
    legend.position = 'none'
  )

# Muestra el gráfico
print(puntos)

# ------------------------------------------
# PUNTO 3: ENTRENAMIENTO DE MODELOS
# ------------------------------------------
X_matrix <- as.matrix(X)
Y_factor <- as.factor(data$label)
Y_matrix <- as.matrix(Y)  # ← Corrección esencial

# a) Modelo sin neuronas ocultas
modelo_0 <- multinom(Y_factor ~ ., data = X)

# b) Modelos con 2, 5 y 15 neuronas, sin regularización
modelo_2 <- nnet(x = X_matrix, y = Y_matrix, size = 2, softmax = TRUE, maxit = 1000)
modelo_5 <- nnet(x = X_matrix, y = Y_matrix, size = 5, softmax = TRUE, maxit = 1000)
modelo_15 <- nnet(x = X_matrix, y = Y_matrix, size = 15, softmax = TRUE, maxit = 1000)

# c) Modelos con 15 neuronas y regularización
modelo_15_d01 <- nnet(x = X_matrix, y = Y_matrix, size = 15, softmax = TRUE, maxit = 1000, decay = 0.1)
modelo_15_d05 <- nnet(x = X_matrix, y = Y_matrix, size = 15, softmax = TRUE, maxit = 1000, decay = 0.5)
modelo_15_d1  <- nnet(x = X_matrix, y = Y_matrix, size = 15, softmax = TRUE, maxit = 1000, decay = 1.0)

# ------------------------------------------
# PUNTO 4: PREDICCIÓN SOBRE LA MALLA (GRID)
# ------------------------------------------
hs <- 0.05  # Reducir la resolución de la malla
grid <- as.matrix(expand.grid(
  x1 = seq(x1_min, x1_max, by = hs),
  x2 = seq(x2_min, x2_max, by = hs)
))

# Modelo sin neuronas ocultas (multinom)
pred_0 <- predict(modelo_0, newdata = as.data.frame(grid))

# Modelos con neuronas ocultas
pred_2 <- as.factor(apply(predict(modelo_2, grid), 1, which.max))
pred_5 <- as.factor(apply(predict(modelo_5, grid), 1, which.max))
pred_15 <- as.factor(apply(predict(modelo_15, grid), 1, which.max))

pred_15_d01 <- as.factor(apply(predict(modelo_15_d01, grid), 1, which.max))
pred_15_d05 <- as.factor(apply(predict(modelo_15_d05, grid), 1, which.max))
pred_15_d1  <- as.factor(apply(predict(modelo_15_d1,  grid), 1, which.max))

# ------------------------------------------
# PUNTO 5: FUNCIÓN PARA VISUALIZAR MODELOS
# ------------------------------------------
plot_model <- function(prediction, grid, title, file_name) {
  df_grid <- data.frame(grid, class = prediction)
  
  plot <- ggplot() +
    geom_tile(data = df_grid, aes(x = x1, y = x2, fill = class), alpha = 0.3) +
    geom_point(data = data, aes(x = x1, y = x2, color = as.factor(label)), size = 1.2) +
    theme_bw(base_size = 14) +
    coord_fixed() +
    labs(title = title) +
    theme(
      legend.position = "none",
      axis.title = element_blank(),
      axis.text = element_blank(),
      axis.ticks = element_blank()
    )
  
  # Guardar la gráfica en un archivo
  ggsave(filename = file_name, plot = plot, width = 8, height = 6, dpi = 300)
}

# Crear una carpeta para guardar las gráficas (si no existe)
dir.create("graficas_modelos", showWarnings = FALSE)

# ------------------------------------------
# PUNTO 6: VISUALIZAR Y GUARDAR LOS MODELOS
# ------------------------------------------
# Guardar las gráficas de los modelos entrenados
plot_model(pred_0, grid, "Modelo sin Neuronas Ocultas", "graficas_modelos/modelo_0.png")
plot_model(pred_2, grid, "Modelo con 2 Neuronas Ocultas", "graficas_modelos/modelo_2.png")
plot_model(pred_5, grid, "Modelo con 5 Neuronas Ocultas", "graficas_modelos/modelo_5.png")
plot_model(pred_15, grid, "Modelo con 15 Neuronas Ocultas", "graficas_modelos/modelo_15.png")
plot_model(pred_15_d01, grid, "Modelo con 15 Neuronas + Decay = 0.1", "graficas_modelos/modelo_15_d01.png")
plot_model(pred_15_d05, grid, "Modelo con 15 Neuronas + Decay = 0.5", "graficas_modelos/modelo_15_d05.png")
plot_model(pred_15_d1, grid, "Modelo con 15 Neuronas + Decay = 1.0", "graficas_modelos/modelo_15_d1.png")
