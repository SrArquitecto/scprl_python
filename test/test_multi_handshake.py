"""
Test del handshake MULTI_INIT_ del PettingZoo mode en ControlServer.

Uso:
  1) Compilar y arrancar el plugin de SCP:SL con el .dll nuevo
  2) En otra terminal, ejecutar este script
  3) El script hace:
     a) Conecta al puerto 7900
     b) Envía "MULTI_INIT_4\n"
     c) Espera respuesta "MULTI_REGISTERED\n"
     d) Envía un dict de acciones: {"agent_0": 5, "agent_1": 12, ...}
     e) Espera respuesta: dict con 4 observaciones
"""
import socket
import sys
import time
import json
import threading

HOST = "localhost"
PORT = 7900
TIMEOUT = 5.0  # segundos para esperar respuesta

def main():
    print(f"=== Test PettingZoo Handshake ===")
    print(f"Conectando a {HOST}:{PORT}...")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(TIMEOUT)
        sock.connect((HOST, PORT))
    except (ConnectionRefusedError, socket.timeout) as e:
        print(f"❌ No se pudo conectar: {e}")
        print("   ¿Está el plugin corriendo y el puerto abierto?")
        sys.exit(1)

    print("✅ Conectado")

    # 1) Handshake
    print("\n[1] Enviando handshake MULTI_INIT_4...")
    sock.sendall(b"MULTI_INIT_4\n")

    # Leer respuesta
    resp = read_line(sock, timeout=3.0)
    if resp is None:
        print("❌ No se recibió respuesta al handshake (timeout)")
        sys.exit(1)
    if resp == "MULTI_REGISTERED":
        print(f"✅ Handshake OK: {resp}")
    else:
        print(f"❌ Respuesta inesperada: {resp!r}")
        sys.exit(1)

    # 2) Loop: enviar dict de acciones, recibir dict de observaciones
    print("\n[2] Loop de prueba: 3 steps")
    for step in range(3):
        # Enviar dict de acciones: cada agent con su action_id (int)
        actions = {
            f"agent_{i}": (i * 3 + step) % 13  # diferentes acciones por agente y step
            for i in range(4)
        }
        actions_json = json.dumps(actions)
        print(f"  Step {step}: enviando {actions_json[:80]}...")
        sock.sendall(actions_json.encode() + b"\n")

        # Leer respuesta (dict combinado de observaciones)
        resp = read_line(sock, timeout=3.0)
        if resp is None:
            print(f"  ❌ Step {step}: timeout esperando obs dict")
            break
        try:
            obs_dict = json.loads(resp)
            if not isinstance(obs_dict, dict):
                print(f"  ❌ Step {step}: respuesta no es dict: {type(obs_dict)}")
                break
            keys = sorted(obs_dict.keys())
            print(f"  ✅ Step {step}: obs dict con {len(obs_dict)} agentes: {keys}")
            # Verificar que cada obs es JSON string
            for k, v in obs_dict.items():
                if not isinstance(v, str):
                    print(f"  ❌ Step {step}: {k} no es string, es {type(v)}")
                    break
                # Intentar parsear el obs como JSON
                try:
                    json.loads(v)
                except json.JSONDecodeError as e:
                    print(f"  ❌ Step {step}: {k} no es JSON válido: {e}")
                    break
            else:
                # Mostrar tamaño de cada obs
                for k, v in obs_dict.items():
                    print(f"     {k}: {len(v)} chars")
                continue
            break
        except json.JSONDecodeError as e:
            print(f"  ❌ Step {step}: respuesta no es JSON válido: {e}")
            print(f"     Primera parte: {resp[:200]}")
            break

        time.sleep(0.1)

    print("\n[3] Cerrando conexión...")
    sock.close()
    print("✅ Test completado")


def read_line(sock, timeout):
    """Lee hasta \n o timeout."""
    sock.settimeout(timeout)
    data = b""
    try:
        while True:
            chunk = sock.recv(1)
            if not chunk:
                return None
            data += chunk
            if chunk == b"\n":
                return data.decode("utf-8-sig").strip()
    except socket.timeout:
        return None
    except Exception as e:
        print(f"  Error leyendo: {e}")
        return None


if __name__ == "__main__":
    main()
