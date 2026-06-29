
from scp_env.reward_manager.rewards.fase_1.reward_f1_1 import Fase1SalirHabitacion

class RewardManager():
    def __init__(self):
        # Mapeamos los niveles de currículum a sus clases evaluadoras
        self.fases = {
            1: Fase1SalirHabitacion(),
            #2: Fase2Exploracion(),
            # 3: Fase3KeycardsAndLockers(), etc.
        }
        self.fase_actual = 1
        
        # Métricas para subir de nivel automáticamente
        self.conteo_episodios = 0
        self.exitos_consecutivos = 0

    def reset_fase_actual(self):
        self.fases[self.fase_actual].reset()

    def calcular_reward(self, s: dict, action: int, info_entorno: dict, pasos) -> tuple[float, bool]:
        # Llama dinámicamente a la estrategia que toca
        return self.fases[self.fase_actual].calcular_reward(s, action, info_entorno, pasos)

    def evaluar_desempeno(self, info_fin_episodio: dict):
        """
        Se ejecuta al final de cada episodio (cuando terminated o truncated es True).
        Determina si el agente está listo para pasar a la siguiente fase de recompensas.
        """
        self.conteo_episodios += 1
        
        # Ejemplo de KPI: ¿Logró salir de la habitación en este episodio?
        salio_con_exito = info_fin_episodio.get("LeftSpawnRoom", False)
        
        if self.fase_actual == 1:
            if salio_con_exito:
                self.exitos_consecutivos += 1
            else:
                self.exitos_consecutivos = max(0, self.exitos_consecutivos - 1)
                
            # Si logra salir con éxito 15 veces seguidas, sube de nivel
            if self.exitos_consecutivos >= 15:
                self.fase_actual = 2
                self.exitos_consecutivos = 0
                print(f"🌟 [REWARD MANAGER] ¡Currículum aumentado a FASE 2! El agente domina la salida.")