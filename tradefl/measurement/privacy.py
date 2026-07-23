"""Configurable privacy surface scoring."""
def privacy_surface(information_type_risk:float, transmission_frequency_risk:float, recipient_risk:float, protection_factor:float)->float:
    return information_type_risk*transmission_frequency_risk*recipient_risk*protection_factor
