"""Agente Constructor: Diseña y refactoriza candidatos de código protegidos para El Coliseo."""
class Constructor:
    def __init__(self):
        self.generation_count = 0

    def generate_candidate_kalman_step(self, hardened: bool = True):
        """Genera una función candidata de cálculo de innovación y covarianza."""
        self.generation_count += 1
        if hardened:
            def hardened_step(p_val: float, r_val: float):
                # Candidato endurecido con protección contra división por cero y varianzas negativas
                safe_r = max(float(r_val), 1e-6)
                safe_p = max(float(p_val), 0.0)
                denom = safe_p + safe_r
                k_gain = safe_p / denom
                p_next = (1.0 - k_gain) * safe_p
                return k_gain, p_next
            return hardened_step
        else:
            def naive_step(p_val, r_val):
                # Candidato ingenuo vulnerable a NaNs y ZeroDivision
                k_gain = p_val / (p_val + r_val)
                p_next = (1.0 - k_gain) * p_val
                return k_gain, p_next
            return naive_step
