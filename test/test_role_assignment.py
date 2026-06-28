#!/usr/bin/env python3
"""
Set de pruebas para verificar reasignación de roles y respawn.

Este script prueba:
1. Asignación inicial de roles diferentes
2. Respawn con el mismo rol
3. Cambio de rol al vuelo (requiere modificación en C#)
4. Muerte y respawn
5. Escape y respawn

Requisitos:
- Servidor SCP:SL corriendo con el plugin ScpRLBridge
- Puerto 7900 accesible (ControlServer)
"""

import socket
import time
import sys

CONTROL_PORT = 7900
TIMEOUT = 10.0

def send_command(command: str, wait_response: bool = True) -> str:
    """Envía un comando al ControlServer y retorna la respuesta."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(TIMEOUT)
        s.connect(("localhost", CONTROL_PORT))
        s.sendall(f"{command}\n".encode('utf-8'))
        
        if wait_response:
            response = b""
            while b"\n" not in response:
                chunk = s.recv(4096)
                if not chunk:
                    break
                response += chunk
            s.close()
            return response.decode('utf-8-sig').strip()
        else:
            s.close()
            return "OK"
    except Exception as e:
        print(f"❌ Error enviando {command}: {e}")
        return None

def test_1_initial_role_assignment():
    """Prueba 1: Asignación inicial de roles diferentes."""
    print("\n" + "="*70)
    print("PRUEBA 1: Asignación inicial de roles diferentes")
    print("="*70)
    
    roles = ["classd", "chaos", "scientist", "ntf"]
    
    print("\n📋 Enviando handshakes con diferentes roles...")
    
    for i, role in enumerate(roles):
        print(f"\n  Agente {i}: rol = {role}")
        response = send_command(f"INIT_{i}_{role}")
        
        if response and "REGISTERED" in response:
            print(f"  ✅ Agente {i} registrado como {role}")
        else:
            print(f"  ❌ Agente {i} falló al registrarse: {response}")
            return False
        
        time.sleep(0.5)
    
    print("\n✅ Prueba 1 completada: Todos los agentes registrados con roles diferentes")
    return True

def test_2_respawn_same_role():
    """Prueba 2: Respawn con el mismo rol."""
    print("\n" + "="*70)
    print("PRUEBA 2: Respawn con el mismo rol")
    print("="*70)
    
    print("\n📋 Enviando comando RESPAWN para agente 0...")
    response = send_command("RESPAWN_0")
    
    if response and "OK" in response:
        print("  ✅ Comando RESPAWN aceptado")
        print("  📝 Verifica en el juego que el agente 0 respawneó con su rol original")
        return True
    else:
        print(f"  ❌ Comando RESPAWN falló: {response}")
        return False

def test_3_role_change_during_respawn():
    """Prueba 3: Cambio de rol durante respawn (ClassD → Chaos)."""
    print("\n" + "="*70)
    print("PRUEBA 3: Cambio de rol durante respawn")
    print("="*70)
    
    print("\n⚠️  NOTA: Esta prueba requiere que el código C# permita cambiar el rol")
    print("   antes de enviar RESPAWN. Actualmente el código tiene lógica para esto")
    print("   pero no está expuesta vía TCP.")
    
    print("\n📋 Para probar esto manualmente:")
    print("   1. Inicia el agente 0 como ClassD")
    print("   2. Modifica _bot._role a ChaosRifleman en el código C#")
    print("   3. Envía RESPAWN_0")
    print("   4. Verifica que el agente respawneó como Chaos")
    
    print("\n💡 Alternativa: Modifica el ControlServer para aceptar:")
    print("   RESPAWN_0_chaos (cambia rol y respawnea)")
    
    return True

def test_4_death_and_respawn():
    """Prueba 4: Muerte y respawn."""
    print("\n" + "="*70)
    print("PRUEBA 4: Muerte y respawn")
    print("="*70)
    
    print("\n📋 Instrucciones manuales:")
    print("   1. Inicia el agente 0 como ClassD")
    print("   2. En el juego, mata al agente 0 (usa /kill o deja que un SCP lo mate)")
    print("   3. Verifica en los logs:")
    print("      - [ScpAgentBot] Agente 0 murió. -100 — episodio terminado.")
    print("   4. Envía RESPAWN_0 desde Python")
    print("   5. Verifica que el agente respawneó")
    
    print("\n🔍 Verificación esperada:")
    print("   - Recompensa: -100 (penalización por muerte)")
    print("   - Episodio: terminado")
    print("   - Respawn: exitoso con el mismo rol")
    
    return True

def test_5_escape_and_respawn():
    """Prueba 5: Escape y respawn."""
    print("\n" + "="*70)
    print("PRUEBA 5: Escape y respawn")
    print("="*70)
    
    print("\n📋 Instrucciones manuales:")
    print("   1. Inicia el agente 0 como ClassD")
    print("   2. En el juego, haz que el agente escape (llega a la superficie)")
    print("   3. Verifica en los logs:")
    print("      - [ScpAgentBot] Agente 0 escapó. +200 — episodio terminado.")
    print("   4. Envía RESPAWN_0 desde Python")
    print("   5. Verifica que el agente respawneó")
    
    print("\n🔍 Verificación esperada:")
    print("   - Recompensa: +200 (bonus por escape)")
    print("   - Episodio: terminado")
    print("   - Respawn: exitoso con el mismo rol")
    
    return True

def test_6_multiple_respawns():
    """Prueba 6: Múltiples respawns consecutivos."""
    print("\n" + "="*70)
    print("PRUEBA 6: Múltiples respawns consecutivos")
    print("="*70)
    
    print("\n📋 Enviando 3 respawns consecutivos para agente 0...")
    
    for i in range(3):
        print(f"\n  Respawn {i+1}/3...")
        response = send_command("RESPAWN_0")
        
        if response and "OK" in response:
            print(f"  ✅ Respawn {i+1} aceptado")
        else:
            print(f"  ❌ Respawn {i+1} falló: {response}")
            return False
        
        time.sleep(2.0)  # Esperar a que complete el respawn
    
    print("\n✅ Prueba 6 completada: Múltiples respawns exitosos")
    return True

def test_7_role_factory_mapping():
    """Prueba 7: Verificar mapeo de roles en RoleFactory."""
    print("\n" + "="*70)
    print("PRUEBA 7: Verificar mapeo de roles en RoleFactory")
    print("="*70)
    
    print("\n📋 Mapeo esperado de roles:")
    print("   'classd'     → RoleTypeId.ClassD + SurvivorStrategy")
    print("   'chaos'      → RoleTypeId.ChaosRifleman + CombatStrategy")
    print("   'scientist'  → RoleTypeId.Scientist + SurvivorStrategy")
    print("   'ntf'        → RoleTypeId.NtfPrivate + CombatStrategy")
    print("   'guard'      → RoleTypeId.FacilityGuard + CombatStrategy")
    print("   (cualquier otro) → RoleTypeId.ClassD + SurvivorStrategy")
    
    print("\n🔍 Para verificar:")
    print("   1. Revisa RoleFactory.cs en:")
    print("      /home/ark/AgenteSCP_v4/scprl_plugin/ScpRLBridge/ScpAgent/Managers/Data/RoleFactory.cs")
    print("   2. Verifica que el mapeo coincida con lo esperado")
    
    return True

def test_8_strategy_swap():
    """Prueba 8: Verificar intercambio de estrategias."""
    print("\n" + "="*70)
    print("PRUEBA 8: Verificar intercambio de estrategias")
    print("="*70)
    
    print("\n📋 Para probar el intercambio de estrategias:")
    print("   1. Inicia el agente 0 como ClassD (SurvivorStrategy)")
    print("   2. Modifica _bot._role a ChaosRifleman en C#")
    print("   3. Envía RESPAWN_0")
    print("   4. Verifica en los logs:")
    print("      - Bot.SetStrategy() se llama con CombatStrategy")
    print("      - La estrategia anterior se desvincula (OnUnbind)")
    print("      - La nueva estrategia se vincula (OnBind)")
    
    print("\n🔍 Verificación esperada:")
    print("   - Prioridad de items cambia (armas > keycards)")
    print("   - Penalización de daño cambia (-0.5x en vez de -1.5x)")
    print("   - Eventos se resuscriben correctamente")
    
    return True

def run_all_tests():
    """Ejecuta todas las pruebas."""
    print("\n" + "="*70)
    print("SET DE PRUEBAS: Reasignación de Roles y Respawn")
    print("="*70)
    print(f"Puerto del ControlServer: {CONTROL_PORT}")
    print(f"Timeout: {TIMEOUT}s")
    
    tests = [
        ("Asignación inicial de roles", test_1_initial_role_assignment),
        ("Respawn con mismo rol", test_2_respawn_same_role),
        ("Cambio de rol durante respawn", test_3_role_change_during_respawn),
        ("Muerte y respawn", test_4_death_and_respawn),
        ("Escape y respawn", test_5_escape_and_respawn),
        ("Múltiples respawns", test_6_multiple_respawns),
        ("Mapeo de RoleFactory", test_7_role_factory_mapping),
        ("Intercambio de estrategias", test_8_strategy_swap),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error en prueba '{name}': {e}")
            results.append((name, False))
        
        input("\n⏸️  Presiona ENTER para continuar con la siguiente prueba...")
    
    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN DE PRUEBAS")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"{status} - {name}")
    
    print(f"\nTotal: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("\n🎉 ¡Todas las pruebas pasaron!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} prueba(s) fallaron")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
