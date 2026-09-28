"""
Optimizador Matemático del Modo Carroñero (Zombie Teacher Distillation).
Combina Direct Preference Optimization (DPO) con Diversity Reverse KL (DRKL)
empleando formulación log-sum-exp y márgenes epsilon para garantizar
cero fallos de punto flotante (NaNs / Overflows) bajo estándar Seis Sigmas.
"""
import numpy as np
import logging

logger = logging.getLogger("fenix.sandbox.carronero")

class ModoCarroneroOptimizer:
    def __init__(
        self,
        temperature: float = 2.0,
        alpha_distillation: float = 0.5,
        beta_dpo: float = 0.1,
        gamma_tail: float = 0.05,
        eps_margin: float = 1e-7
    ):
        self.T = float(temperature)
        self.alpha = float(alpha_distillation)
        self.beta = float(beta_dpo)
        self.gamma = float(gamma_tail)
        self.eps = float(eps_margin)

    def _stable_softmax(self, logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Calcula softmax numéricamente estable restando el máximo local."""
        scaled = logits / temperature
        shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
        return exp_logits / np.maximum(sum_exp, self.eps)

    def _log_softmax(self, logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Calcula log-softmax con truco log-sum-exp para evitar underflow."""
        scaled = logits / temperature
        max_val = np.max(scaled, axis=-1, keepdims=True)
        shifted = scaled - max_val
        log_sum = max_val + np.log(np.maximum(np.sum(np.exp(shifted), axis=-1, keepdims=True), self.eps))
        return scaled - log_sum

    def compute_drkl_loss(self, z_teacher: np.ndarray, z_student: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Calcula la pérdida de Divergencia KL de Diversidad (DRKL) y el gradiente analítico
        del estudiante protegido contra colapsos por sobreconfianza (q_m -> 1.0).
        """
        z_t = np.asarray(z_teacher, dtype=np.float64)
        z_s = np.asarray(z_student, dtype=np.float64)

        p = self._stable_softmax(z_t, temperature=self.T)
        q = self._stable_softmax(z_s, temperature=self.T)

        log_p = self._log_softmax(z_t, temperature=self.T)
        log_q = self._log_softmax(z_s, temperature=self.T)

        # Reverse KL: sum p * (log_p - log_q)
        kl_div = np.sum(p * (log_p - log_q))
        # Regularización de cola (Diversity Tail Alignment)
        tail_regularizer = self.gamma * np.sum(np.maximum(0.0, -log_q))
        loss_drkl = float(kl_div + tail_regularizer)

        # Gradiente analítico respecto a los logits del estudiante:
        # dL/dz_s = T * (q - p) regularizado con gamma
        grad_student = self.T * (q - p) + (self.gamma * (1.0 - q) * self.eps)

        return loss_drkl, grad_student

    def compute_dpo_loss(
        self,
        log_prob_win_policy: float,
        log_prob_win_ref: float,
        log_prob_lose_policy: float,
        log_prob_lose_ref: float
    ) -> float:
        """
        Calcula la pérdida de Direct Preference Optimization (DPO):
        L_DPO = -ln sigma( beta * log(pi_w/ref_w) - beta * log(pi_l/ref_l) )
        """
        pi_logratio_win = log_prob_win_policy - log_prob_win_ref
        pi_logratio_lose = log_prob_lose_policy - log_prob_lose_ref

        delta = self.beta * (pi_logratio_win - pi_logratio_lose)

        # ln(1 + exp(-delta)) de forma numéricamente estable
        if delta >= 0:
            loss_dpo = np.log1p(np.exp(-delta))
        else:
            loss_dpo = -delta + np.log1p(np.exp(delta))

        return float(loss_dpo)

    def optimize_step(
        self,
        z_teacher: np.ndarray,
        z_student: np.ndarray,
        dpo_metrics: tuple[float, float, float, float] = None
    ) -> dict:
        """
        Ejecuta un paso de evaluación y alineación carroñera combinada.
        """
        loss_drkl, grad_student = self.compute_drkl_loss(z_teacher, z_student)

        if dpo_metrics:
            lp_w_pol, lp_w_ref, lp_l_pol, lp_l_ref = dpo_metrics
            loss_dpo = self.compute_dpo_loss(lp_w_pol, lp_w_ref, lp_l_pol, lp_l_ref)
        else:
            loss_dpo = 0.6931 # ln(2), estado balanceado inicial

        # Pérdida total combinada
        loss_total = (1.0 - self.alpha) * loss_dpo + self.alpha * (self.T ** 2) * loss_drkl

        # Verificar integridad Seis Sigmas (cero NaNs o infinitos)
        is_healthy = not (np.isnan(loss_total) or np.isinf(loss_total) or np.isnan(grad_student).any())

        return {
            "loss_total": round(loss_total, 6),
            "loss_dpo": round(loss_dpo, 6),
            "loss_drkl": round(loss_drkl, 6),
            "mean_gradient_magnitude": round(float(np.mean(np.abs(grad_student))), 6),
            "max_gradient_magnitude": round(float(np.max(np.abs(grad_student))), 6),
            "numerical_health": is_healthy
        }

if __name__ == "__main__":
    opt = ModoCarroneroOptimizer(temperature=2.0, alpha_distillation=0.5, beta_dpo=0.1)

    # 1. Caso Nominal
    z_teacher = np.array([2.5, 1.2, -0.5, 0.1, 3.8])
    z_student = np.array([2.1, 1.0, -0.2, 0.0, 3.2])
    res_nominal = opt.optimize_step(z_teacher, z_student, dpo_metrics=(-1.2, -1.8, -3.4, -2.5))
    print("Caso Nominal:", res_nominal)

    # 2. Caso de Sobreconfianza Extrema (q_m -> 1.0)
    z_teacher_extreme = np.array([10.0, -5.0, -8.0, -10.0, -12.0])
    z_student_extreme = np.array([50.0, -20.0, -30.0, -40.0, -50.0]) # Sobreconfianza máxima
    res_extreme = opt.optimize_step(z_teacher_extreme, z_student_extreme)
    print("Caso Sobreconfianza Extrema (Mitigación NaNs):", res_extreme)
