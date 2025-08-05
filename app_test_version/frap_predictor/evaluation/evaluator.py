import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

class ModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calcula y retorna todas las métricas relevantes"""
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred)
        }

    @staticmethod
    def plot_predictions(y_true, y_pred, title="Predicciones vs Valores Reales"):
        """Genera gráfico de comparación entre predicciones y valores reales"""
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_true, y=y_pred)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
        plt.xlabel('Valores Reales')
        plt.ylabel('Predicciones')
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_training_history(history):
        """Visualiza el historial de entrenamiento del modelo"""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Evolución de la Pérdida')
        plt.xlabel('Época')
        plt.ylabel('MSE')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Train MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Evolución del MAE')
        plt.xlabel('Época')
        plt.ylabel('MAE')
        plt.legend()

        plt.tight_layout()
        plt.show()
