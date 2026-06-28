from turtle import pos
import time
import socket
import time
import orjson


class Comunicacion:

    def __init__(self, host = "localhost", port = 7900, agent_id = 0, role = "classd"):
        self.agent_id = agent_id
        self.host = host
        self.port = port
        self.sock = None
        self.role = role
        
    def conexion(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Desactivar el algoritmo de Nagle para enviar/recibir datos instantáneamente sin agrupar
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(10.0)  # timeout para no bloquear infinitamente si el plugin no responde
        self.sock.connect((self.host, self.port))
        # Enviar Handshake
        self.sock.sendall(f"INIT_{self.agent_id}_{self.role}\n".encode('utf-8'))

        # Crear un archivo virtual para usar readline de forma segura
        sock_file = self.sock.makefile('r', encoding='utf-8')

        resp = ""
        for intento in range(10):
            resp = sock_file.readline().strip()
            if resp == "REGISTERED":
                print(f"✅ Agente {self.agent_id} conectado con éxito en el intento {intento + 1}.")
                return
            elif resp == "":
                # El buffer está vacío, esperamos un instante a que llegue el paquete de red
                time.sleep(0.05)
                continue
            else:
                break

    def enviar_datos(self, data):
        try:
            self.sock.sendall(data.encode())
        except (BrokenPipeError, ConnectionResetError):
            print("⚠️ Conexión perdida. Intentando reconectar...")
            self.conexion() # Reconecta automáticamente
            self.sock.sendall(data.encode())

    def enviar_accion(self, a):
        """
        Ahora solo envía el índice de acción al servidor TCP.
        La ejecución física ocurre en el C# (StateManager.cs).
        """
        msg = f"ACTION:{a}\n"
        return self.enviar_solicitud(msg.encode('utf-8'))
        #self.sock.sendall(msg.encode('utf-8'))
        
    def solicitar_estado(self, retries=5):
        #print("1")
        try:
            resultado = self.enviar_solicitud(b"GET_STATE\n")
            #print("1.5 - Respuesta recibida con éxito")
            #print(resultado)
            return resultado
        except Exception as e:
            print(f"1.5 - ERROR CRÍTICO EN REQUEST: {e}")
            raise e
        
    def respawn(self):
        try:
            self.enviar_solicitud(b"RESPAWN\n")
        except Exception:
            pass
        
    def reconectar(self, max_retries=60, retry_delay=3.0):
        """Cierra el socket viejo y reconecta usando RECONNECT_.
        El plugin responde REGISTERED si el bucle maestro está activo, o WAIT si está parado.
        """
        if self.sock is not None:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None

        for attempt in range(1, max_retries + 1):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock.settimeout(10.0)
                self.sock.connect((self.host, self.port))

                self.sock.sendall(f"RECONNECT_{self.agent_id}\n".encode('utf-8'))

                # utf-8-sig elimina el BOM (\uFEFF) si el plugin lo emite por error
                sock_file = self.sock.makefile('r', encoding='utf-8-sig')
                resp = ""
                for _ in range(10):
                    resp = sock_file.readline().strip()
                    if resp:
                        break
                    time.sleep(0.05)

                if resp == "REGISTERED":
                    print(f"✅ Agente {self.agent_id} reconectado (intento {attempt}/{max_retries}).")
                    return True
                elif resp == "WAIT":
                    self.sock.close()
                    self.sock = None
                    if attempt % 5 == 0:
                        print(f"⏳ Agente {self.agent_id} plugin en pausa, esperando... (intento {attempt}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                elif resp == "DUPLICATE":
                    self.sock.close()
                    self.sock = None
                    print(f"⚠️ Agente {self.agent_id} ya está registrado. Esperando cierre...")
                    time.sleep(retry_delay)
                    continue
                else:
                    self.sock.close()
                    self.sock = None
                    print(f"⚠️ Agente {self.agent_id} respuesta inesperada: {repr(resp)}. Reintentando...")
                    time.sleep(retry_delay)
                    continue
            except Exception as e:
                print(f"⏳ Agente {self.agent_id} reconexión falló (intento {attempt}/{max_retries}): {e}")
                try: self.sock.close()
                except: pass
                self.sock = None
                time.sleep(retry_delay)

        print(f"❌ Agente {self.agent_id} no pudo reconectar tras {max_retries} intentos.")
        return False

    def enviar_solicitud(self, message: bytes, retries=5, delay=1.0):
        t0 = time.perf_counter()
        for attempt in range(retries):
            try:
                self.sock.sendall(message)
                buf = b""
                while b"\n" not in buf:
                    chunk = self.sock.recv(4096)
                    if not chunk:
                        raise RuntimeError("Conexión perdida (EOF)")
                    buf += chunk
                line = buf.split(b"\n")[0].decode("utf-8-sig").strip()
                result = orjson.loads(line)
                elapsed = (time.perf_counter() - t0) * 1000
                return result

            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                print(f"⚠️ Agente {self.agent_id} conexión rota en intento {attempt + 1}/{retries}: {e}")
                if attempt < retries - 1:
                    if self.reconectar():
                        continue
                    else:
                        break
                raise
            except Exception as e:
                print(f"⚠️ Agente {self.agent_id} error en request (intento {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                    continue
                raise
    
    def cerrar(self):
        self.sock.close()
    