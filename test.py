 
import sys
import traceback
import numpy as np
import os
import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback )
# ====== 1. IMPORTA TU ENTORNO ======
# Cambia 'nombre_de_tu_archivo' por el nombre real de tu script de entorno
# y 'SCPEnv' por el nombre de tu clase.
try:
    from scp_env2 import SCPClassDEnv  # 👈 Cambia esto si tu archivo/clase se llaman diferente
    print("✅ Archivo de entorno importado correctamente.")
except Exception as e:
    print("❌ ERROR CRÍTICO al importar el entorno:")
    traceback.print_exc()
    sys.exit(1)

# ====== 2. FUNCIÓN DE PRUEBA ======
def run_diagnostic():
    print("\n--- INICIANDO DIAGNÓSTICO DEL ENTORNO ---")
    
    # Instanciar el entorno
    try:
        print("[1/4] Instanciando la clase del entorno...")
        env = SCPClassDEnv()
        print("✅ Clase instanciada con éxito.")
    except Exception as e:
        print("❌ ERROR en el __init__ de tu entorno:")
        traceback.print_exc()
        return

    # Probar el Reset (Conexión con el juego)
    try:
        print("\n[2/4] Intentando ejecutar env.reset()... (Esperando al juego)")
        obs = env.reset()
        print("✅ env.reset() ejecutado correctamente.")
        print(f"   -> Tipo de observación devuelta: {type(obs)}")
        if isinstance(obs, tuple):
            print(f"   -> Parece que devuelve (obs, info). Tipo de la obs: {type(obs[0])}")
            obs_to_check = obs[0]
        else:
            obs_to_check = obs
            
        print(f"   -> Shape de la observación: {getattr(obs_to_check, 'shape', 'No tiene shape')}")
    except Exception as e:
        print("❌ ERROR en el env.reset() [Posible fallo de conexión TCP/JSON]:")
        traceback.print_exc()
        return

    # Probar un Step manual
    try:
        print("\n[3/4] Intentando ejecutar env.step()...")
        # Creamos una acción dummy (ajusta el tamaño si tu action_space es diferente)
        # Si tu espacio es discreto, pon un entero (ej: 0). Si es continuo, un array de ceros.
        if hasattr(env, 'action_space'):
            print(f"   -> Tu action_space es: {env.action_space}")
            dummy_action = env.action_space.sample()
            print(f"   -> Usando acción aleatoria de prueba: {dummy_action}")
        else:
            dummy_action = np.array([0.0, 0.0], dtype=np.float32)
            print(f"   -> No se detectó action_space, usando por defecto: {dummy_action}")

        # Ejecutamos el step
        resultado = env.step(dummy_action)
        print("✅ env.step() ejecutado correctamente.")
        
        # Desempaquetar según la versión de Gym/Gymnasium
        if len(resultado) == 5:
            obs, reward, terminated, truncated, info = resultado
            print("   -> Formato Gymnasium (5 valores de retorno) detectado.")
        elif len(resultado) == 4:
            obs, reward, done, info = resultado
            print("   -> Formato Gym clásico (4 valores de retorno) detectado.")
        
    except Exception as e:
        print("❌ ERROR en el env.step() [Aquí es donde probablemente se rompe la normalización]:")
        traceback.print_exc()
        return

    # Verificar el array 'base' final
    try:
        print("\n[4/4] Verificando la consistencia de los datos del estado...")
        # Intentamos forzar la lectura de las variables que nos interesan si están accesibles
        print("   -> Estructura del último estado recibido:")
        
        # Comprobación de tipos en el array que se ha generado
        if 'obs' in locals():
            # Si tu observación es un diccionario (Dict space) con 'base' e 'imagen'
            if isinstance(obs, dict):
                print("   -> Tu observación es un diccionario.")
                for k, v in obs.items():
                    print(f"      - Clave '{k}': tipo {type(v)}, shape/longitud: {getattr(v, 'shape', len(v))}")
                    if np.any(np.isnan(v)):
                        print(f"        ⚠️ ¡ALERTA! Contiene valores NaN en la clave {k}")
            else:
                # Si es un array plano u otra cosa
                if np.any(np.isnan(obs)):
                    print("   ⚠️ ¡ALERTA! El vector de observaciones devuelto contiene valores NaN (Not a Number).")
                else:
                    print("   ✅ El vector de observaciones NO contiene NaNs.")
                
                if np.any(np.isinf(obs)):
                    print("   ⚠️ ¡ALERTA! El vector de observaciones contiene valores Infinitos.")
                    
    except Exception as e:
        print("❌ Error al analizar el estado resultante:")
        traceback.print_exc()

    print("\n--- DIAGNÓSTICO FINALIZADO ---")

if __name__ == "__main__":
    run_diagnostic()
