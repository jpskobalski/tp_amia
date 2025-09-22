import numpy as np
import numpy.linalg as LA
from base.qda import TensorizedQDA


class FasterQDA(TensorizedQDA):
    """
    Vectorizamos la prediccion sobre TODAS las observaciones a la vez. Esta versión arma una matriz (k, n, n) por clase y despues soquedamos unicamente con la diagonal.
    """

    def predict(self, X):
        """
        X: array de shape (p, n)  -> p features, n observaciones
        return: array de shape (n,) con la clase predicha por observación
        """

        # 1) Centramos todas las observaciones X con el vector de medias de CADA clase
        #    - self.tensor_means tiene shape (k, p, 1)
        #    - X tiene shape (p, n)
        #    Con broadcasting, obtengo unbiased de shape (k, p, n): para cada clase k, resto su media a todas las columnas de X.
        unbiased = X[None, :, :] - self.tensor_means  # (k, p, n)

        # 2) Aplicamos la forma cuadratica (x - μ)^T Σ^{-1} (x - μ) pero en batches de observaciones:
        #    Primero transponemos unbiased para tener (k, n, p) a la izquierda, y dejamos (k, p, p) y (k, p, n) en el medio y derecha.
        #    El producto da M con shape (k, n, n): ahi aparecen TODAS las interacciones entre observaciones.
        m = np.transpose(unbiased, (0, 2, 1)) @ self.tensor_inv_cov @ unbiased  # (k, n, n)

        # 3) Para la forma cuadratica solo necesitamps la diagonal por clase/obs:
        #    diag(M) tiene shape (k, n). Nos ahorramos usar toda la matriz M despues.
        inner = np.diagonal(m, axis1=1, axis2=2)  # (k, n)

        # 4) terminamos log-determinante para cada clase:
        #    uso slogdet por estabilidad numérica.
        #    Ojo: tengo Σ^{-1} ya precalculada; log|Σ^{-1}| = -log|Σ|.
        #    slogdet sobre (k, p, p) me devuelve arrays de largo k.
        sign, logdet_inv = LA.slogdet(self.tensor_inv_cov)  # (k,)

        # 5) armamos los log-condicionales por clase y por observacion:
        #    broadcasting para que el logdet_inv (k,) se expanda a (k, n).
        log_conditionals = 0.5 * logdet_inv[:, None] - 0.5 * inner  # (k, n)

        # 6) sumamos log-priors por clase y elejimos la clase con mayor puntaje para cada obs
        scores = self.log_a_priori[:, None] + log_conditionals  # (k, n)
        yhat = np.argmax(scores, axis=0)  # (n,)
        return yhat

    # Dejo un helper opcional si quiero reusar la parte de log-condicionales en otros métodos
    def _predict_log_conditionals_batch(self, X):
        """
        Devolvemos la matriz (k, n) con los log-condicionales para todas las observaciones.
        Solo lo usamos si queremos calcular proba/scores sin repetir codigo.
        """
        unbiased = X[None, :, :] - self.tensor_means # (k, p, n)
        m = np.transpose(unbiased, (0, 2, 1)) @ self.tensor_inv_cov @ unbiased # (k, n, n)
        inner = np.diagonal(m, axis1=1, axis2=2) # (k, n)
        _, logdet_inv = LA.slogdet(self.tensor_inv_cov) # (k,)
        return 0.5 * logdet_inv[:, None] - 0.5 * inner # (k, n)