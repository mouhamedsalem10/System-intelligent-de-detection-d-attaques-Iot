


def decide(prob_attack: float,
           low: float = 0.30,
           high: float = 0.80) -> dict:
    
    if prob_attack < low:
        return {
            "level"       : "LOW",
            "label"       : "normal",
            "color"       : "green",
            "action"      : "Aucune intervention",
            "description" : "Trafic normal détecté"
        }
    elif prob_attack < high:
        return {
            "level"       : "MEDIUM",
            "label"       : "suspicious",
            "color"       : "orange",
            "action"      : "Surveillance renforcée",
            "description" : "Comportement suspect détecté"
        }
    else:
        return {
            "level"       : "HIGH",
            "label"       : "attack",
            "color"       : "red",
            "action"      : "Isolation recommandée",
            "description" : "Attaque détectée avec haute confiance"
        }