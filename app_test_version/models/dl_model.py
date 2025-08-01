import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from ..utils.logger import get_logger

logger = get_logger(__name__)

class FRAPDeepLearningModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self._build_model()
        self.scaler = None

    def _build_model(self):
        """Construye la arquitectura de la red neuronal"""
        model = Sequential([
            Dense(32, activation='relu', input_shape=(self.input_dim,)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )

        return model

    def train(self, X_train, y_train, validation_data=None, epochs=200, batch_size=16):
        """Entrena el modelo con callbacks"""
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.1, patience=10)
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def evaluate(self, X_test, y_test):
        """Evalúa el modelo en datos de prueba"""
        return self.model.evaluate(X_test, y_test, verbose=0)

    def predict(self, X):
        """Realiza predicciones"""
        return self.model.predict(X).flatten()

    def save(self, model_path, scaler_path=None):
        """Guarda el modelo y el scaler"""
        self.model.save(model_path)
        if scaler_path and hasattr(self, 'scaler'):
            import joblib
            joblib.dump(self.scaler, scaler_path)

    @classmethod
    def load(cls, model_path, scaler_path=None):
        """Carga un modelo guardado"""
        model = tf.keras.models.load_model(model_path)
        dl_model = cls(model.input_shape[1])
        dl_model.model = model

        if scaler_path:
            import joblib
            dl_model.scaler = joblib.load(scaler_path)

        return dl_model
