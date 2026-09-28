"""
Estimador y Predictor de Estados de Memoria mediante Filtro de Kalman Lineal Cuadrático (LQE).
Implementa la formulación estabilizada de Joseph para garantizar covarianza simétrica
definida positiva y acotamiento espectral para estabilidad asintótica estricta.
"""
import numpy as np
import logging

logger = logging.getLogger("fenix.routing.kalman")

class KalmanStateEstimator:
    def __init__(
        self,
        num_agents: int = 4,
        process_noise_q: float = 1e-4,
        sensor_noise_r: float = 1e-2,
        spectral_radius_limit: float = 0.95
    ):
        self.N = num_agents
        self.spectral_radius_limit = spectral_radius_limit

        # Vector de estado inicial (equiprobable)
        self.x = np.ones((self.N, 1), dtype=np.float64) / self.N

        # Covarianza de error inicial
        self.P = np.eye(self.N, dtype=np.float64) * 0.1

        # Matriz de transición de estados A con inercia funcional (diagonal = 0.7, dispersión = 0.3/(N-1))
        off_diag = 0.3 / max(self.N - 1, 1)
        self.A = np.full((self.N, self.N), off_diag, dtype=np.float64)
        np.fill_diagonal(self.A, 0.7)
        self._enforce_spectral_radius()

        # Matriz de acoplamiento de control B
        self.B = np.eye(self.N, dtype=np.float64)

        # Matriz de observación H
        self.H = np.eye(self.N, dtype=np.float64)

        # Covarianzas de ruido Q y R
        self.Q = np.eye(self.N, dtype=np.float64) * process_noise_q
        self.R = np.eye(self.N, dtype=np.float64) * sensor_noise_r

    def _enforce_spectral_radius(self):
        """Acota el radio espectral de A para asegurar estabilidad asintótica estricta."""
        eigenvalues = np.linalg.eigvals(self.A)
        max_eigenval = np.max(np.abs(eigenvalues))
        if max_eigenval > self.spectral_radius_limit:
            scale = self.spectral_radius_limit / (max_eigenval + 1e-8)
            self.A = self.A * scale
            logger.debug(f"Radio espectral ajustado: {max_eigenval:.4f} -> {self.spectral_radius_limit}")

    def predict(self, u=None) -> np.ndarray:
        """
        Fase de Predicción (IMU-step):
        x_hat = A * x + B * u
        P = A * P * A^T + Q
        """
        if u is None:
            u_arr = np.zeros((self.N, 1), dtype=np.float64)
        else:
            u_arr = np.asarray(u, dtype=np.float64)
            if u_arr.ndim == 1:
                u_arr = u_arr.reshape((self.N, 1))

        self.x = self.A @ self.x + self.B @ u_arr
        self.P = self.A @ self.P @ self.A.T + self.Q
        # Garantizar simetría
        self.P = 0.5 * (self.P + self.P.T)
        return self.x

    def update(self, z) -> np.ndarray:
        """
        Fase de Corrección (GPS-step) con Formulación Estabilizada de Joseph:
        K = P * H^T * (H * P * H^T + R)^(-1)
        x = x + K * (z - H * x)
        P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        """
        z_arr = np.asarray(z, dtype=np.float64)
        if z_arr.ndim == 1:
            z_arr = z_arr.reshape((self.N, 1))

        # Innovación
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Actualización de estado
        y = z_arr - (self.H @ self.x)
        self.x = self.x + K @ y

        # Formulación de Joseph para conservar definición positiva y simetría
        I = np.eye(self.N, dtype=np.float64)
        IKH = I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return self.x

    def get_probabilities(self) -> np.ndarray:
        """Retorna probabilidades normalizadas de demanda por agente mediante Softmax numéricamente estable."""
        logits = self.x.flatten()
        shifted = logits - np.max(logits)
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.sum(exp_logits)
        return probs

if __name__ == "__main__":
    kfe = KalmanStateEstimator(num_agents=4)
    print("Probabilidades iniciales:", kfe.get_probabilities())
    kfe.predict(u=[0.0, 1.0, 0.0, 0.0])
    print("Tras predicción con lista:", kfe.get_probabilities())
