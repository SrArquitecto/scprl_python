import socket
import time
from stable_baselines3.common.callbacks import BaseCallback

class MapRegenCallback(BaseCallback):
    REGEN_FLAG = "/tmp/scp_regen_flag"

    def __init__(self, regen_every=1_000, control_port=7900, verbose=0):
        super().__init__(verbose)
        self.regen_every = regen_every
        self.control_port = control_port
        self._last_regen = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_regen >= self.regen_every:
            self._last_regen = self.num_timesteps
            print(f"\n🗺️  Regenerando mapa (timestep {self.num_timesteps})...")

            try:
                s = socket.socket()
                s.settimeout(10.0)
                s.connect(("localhost", self.control_port))
                s.sendall(b"RESTART\n")
                buf = b""
                while b"\n" not in buf:
                    buf += s.recv(4096)
                s.close()
                print("✅ Mapa regenerado.")
            except Exception as e:
                print(f"❌ Error regenerando mapa: {e}")
                return True

            # Bandera para que los workers de SubprocVecEnv trunquen el episodio
            with open(self.REGEN_FLAG, "w") as f:
                f.write(str(self.num_timesteps))

            time.sleep(1.0)

        return True