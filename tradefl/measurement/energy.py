"""Pluggable energy measurement interface; returns unavailable in the prototype."""
class EnergyMeter:
    def start(self)->None: pass
    def stop(self)->float|None: return None
