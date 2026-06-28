import socket
import time

class CurriculumManager:
    def __init__(self, host="localhost", port=8888):
        self.host = host
        self.port = port
 
    def _enviar_comando(self, comando: str, retries=5) -> str:
        for intento in range(retries):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5.0)
                s.connect((self.host, self.port))
                s.sendall(f"{comando}\n".encode())
                buf = b""
                while b"\n" not in buf:
                    buf += s.recv(4096)
                s.close()
                return buf.split(b"\n")[0].decode().strip()
            except Exception as e:
                print(f"⚠️ CurriculumManager intento {intento+1}: {e}")
                time.sleep(1.0)
        return "ERROR"
 
    def set_flag(self, clave: str, valor: bool) -> bool:
        resp = self._enviar_comando(f"CONFIG:{clave}={'true' if valor else 'false'}")
        return resp == "OK"
 
    def set_param(self, clave: str, valor: float) -> bool:
        resp = self._enviar_comando(f"CONFIG:{clave}={valor}")
        return resp == "OK"
 
    def set_fase(self, fase: int) -> bool:
        return self.set_param("FaseActual", fase)
 
    def get_config(self) -> dict:
        import json
        resp = self._enviar_comando("GET_CONFIG")
        try:
            return json.loads(resp)
        except:
            return {}
        
    def fase_1_basico(self):
        """Fase 1: Solo navegación, sin enemigos, sin SCPs, puertas abiertas."""
        self.set_param("FaseActual", 1)
        self.set_param("Dificultad", 0.0)
        self.set_flag("ScpsActivos", False)
        self.set_flag("EnemigosActivos", False)
        self.set_flag("PuertasBloqueadas", False)
        self.set_flag("RespawnInfinito", True)
        self.set_param("NumEnemigos", 0)
        self.set_param("NumScps", 0)
        self.set_param("ProbKeycard", 1.0)
        self.set_param("TiempoMaxEpisodio", 300)
        print("✅ Curriculum Fase 1 activa — navegación básica")
 
    def fase_2_keycards(self):
        """Fase 2: Navegar + recoger keycards, sin enemigos."""
        self.set_param("FaseActual", 2)
        self.set_param("Dificultad", 0.2)
        self.set_flag("PuertasBloqueadas", True)  # ahora necesita keycard
        self.set_param("ProbKeycard", 0.7)         # no garantizada
        self.set_param("TiempoMaxEpisodio", 240)
        print("✅ Curriculum Fase 2 activa — keycards necesarias")
 
    def fase_3_enemigos(self):
        """Fase 3: Navegar + keycards + guardias como amenaza."""
        self.set_param("FaseActual", 3)
        self.set_param("Dificultad", 0.5)
        self.set_flag("EnemigosActivos", True)
        self.set_param("NumEnemigos", 2)
        self.set_flag("RespawnInfinito", False)
        self.set_param("TiempoMaxEpisodio", 180)
        print("✅ Curriculum Fase 3 activa — enemigos presentes")
 
    def fase_4_scps(self):
        """Fase 4: Todo activo, con SCPs."""
        self.set_param("FaseActual", 4)
        self.set_param("Dificultad", 0.8)
        self.set_flag("ScpsActivos", True)
        self.set_param("NumScps", 1)
        self.set_param("NumEnemigos", 2)
        self.set_param("TiempoMaxEpisodio", 120)
        print("✅ Curriculum Fase 4 activa — SCPs presentes")