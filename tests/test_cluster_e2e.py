import unittest
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from enjambre.nodo_maestro import NodoMaestro
from sandbox.constructor import Constructor
from sandbox.disruptor import Disruptor
from sandbox.arbitro import Arbitro
from audit.autopsia_forense import FenixForensicAuditor
from audit.causa_raiz_report import CausaRaizReport
from seguridad.pulsacion_criptografica import PulsacionCriptografica

class TestClusterEndToEnd(unittest.TestCase):
    def setUp(self):
        self.config_path = os.path.join(PROJECT_ROOT, "config.yaml")
        self.maestro = NodoMaestro(self.config_path)
        self.auditor = FenixForensicAuditor()
        self.keystroke = PulsacionCriptografica()

    def tearDown(self):
        self.maestro.paginador.close_all()

    def test_e2e_swarm_and_paging(self):
        # 1. Despachar a Marketing
        res_mkt = self.maestro.dispatch_mission("agente_marketing", {"prompt": "Por favor amablemente optimizar texto"})
        self.assertEqual(res_mkt["status"], "SUCCESS")
        self.assertIn("agente_marketing", res_mkt["resident_in_ram"])

        # 2. Despachar a SRE
        res_sre = self.maestro.dispatch_mission("agente_sre", {})
        self.assertEqual(res_sre["status"], "SUCCESS")
        self.assertTrue(res_sre["data"]["seis_sigmas_compliant"])

        # 3. Despachar a OSINT (desalojando al de menor prioridad por confinamiento de RAM)
        res_osint = self.maestro.dispatch_mission("agente_osint", {"target_host": "localhost"})
        self.assertEqual(res_osint["status"], "SUCCESS")
        self.assertLessEqual(len(self.maestro.paginador.active_pages), 2)

    def test_coliseo_adversarial_workflow(self):
        constructor = Constructor()
        disruptor = Disruptor()
        arbitro = Arbitro()

        # Generar candidato endurecido
        candidate = constructor.generate_candidate_kalman_step(hardened=True)
        stress_cases = disruptor.generate_stress_cases()

        eval_report = arbitro.evaluate_candidate(candidate, stress_cases)
        self.assertGreater(eval_report["passed"], 0)
        self.assertLessEqual(eval_report["failure_rate"], 0.25) # Menos de 1 fallo de estrés

    def test_forensic_autopsy_and_rca(self):
        # Simular captura de fallo
        error_context = {
            "exception_type": "ZeroDivisionError",
            "exception_message": "float division by zero",
            "failing_input": {"denominator": 0.0}
        }
        report = self.auditor.perform_autopsy("agente_marketing", error_context)
        self.assertEqual(report["remediation"]["status"], "IMMUNIZED")
        self.assertEqual(report["remediation"]["fix_applied"], "EPSILON_DENOMINATOR_CLAMP")

        rca_md = CausaRaizReport.generate_markdown_rca(report)
        self.assertIn("REPORTE FORENSE DE CAUSA RAÍZ", rca_md)
        self.assertIn("EPSILON_DENOMINATOR_CLAMP", rca_md)

    def test_keystroke_biometrics(self):
        # Humano normal
        res_legit = self.keystroke.evaluate_keystrokes([110.0, 115.0, 108.0, 112.0, 114.0])
        self.assertTrue(res_legit["authorized"])

        # Script automatizado con variabilidad 0
        res_robot = self.keystroke.evaluate_keystrokes([10.0, 10.0, 10.0, 10.0])
        self.assertFalse(res_robot["authorized"])
        self.assertEqual(res_robot["verdict"], "ROBOTIC_INJECTION")

if __name__ == "__main__":
    unittest.main()
