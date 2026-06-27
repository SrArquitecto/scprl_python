"""
Controlador de input usando evdev/uinput.
Crea dispositivos virtuales de teclado y ratón a nivel de kernel.
El juego los trata como hardware real.
"""
import evdev
from evdev import UInput, ecodes as e
import time

# Teclas necesarias
KEY_MAP = {
    'w':     e.KEY_W,
    's':     e.KEY_S,
    'a':     e.KEY_A,
    'd':     e.KEY_D,
    'e':     e.KEY_E,
    'shift': e.KEY_LEFTSHIFT,
    'ctrl':  e.KEY_LEFTCTRL,
}

class InputController:
    def __init__(self):
        # Capacidades del teclado virtual
        kb_cap = {
            e.EV_KEY: list(KEY_MAP.values())
        }
        # Capacidades del ratón virtual
        mouse_cap = {
            e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT],
            e.EV_REL: [e.REL_X, e.REL_Y],
        }

        self.kb    = UInput(kb_cap,    name="scprl-keyboard", version=0x3)
        self.mouse = UInput(mouse_cap, name="scprl-mouse",    version=0x3)
        time.sleep(0.5)  # esperar a que el SO registre los dispositivos
        print(f"✅ Teclado virtual: {self.kb.name}")
        print(f"✅ Ratón virtual:   {self.mouse.name}")

    def press(self, key, duration=0.1):
        """Pulsar y soltar una tecla."""
        code = KEY_MAP[key]
        self.kb.write(e.EV_KEY, code, 1)  # keydown
        self.kb.syn()
        time.sleep(duration)
        self.kb.write(e.EV_KEY, code, 0)  # keyup
        self.kb.syn()

    def hold(self, key, duration=1.2):
        """Mantener una tecla pulsada."""
        code = KEY_MAP[key]
        self.kb.write(e.EV_KEY, code, 1)
        self.kb.syn()
        time.sleep(duration)
        self.kb.write(e.EV_KEY, code, 0)
        self.kb.syn()

    def press_combo(self, *keys, duration=0.1):
        """Pulsar varias teclas a la vez (ej: shift+w)."""
        codes = [KEY_MAP[k] for k in keys]
        for code in codes:
            self.kb.write(e.EV_KEY, code, 1)
        self.kb.syn()
        time.sleep(duration)
        for code in codes:
            self.kb.write(e.EV_KEY, code, 0)
        self.kb.syn()

    def move_mouse(self, dx, dy):
        """Mover ratón en relativo (movimiento de cámara)."""
        self.mouse.write(e.EV_REL, e.REL_X, dx)
        self.mouse.write(e.EV_REL, e.REL_Y, dy)
        self.mouse.syn()

    def close(self):
        self.kb.close()
        self.mouse.close()
