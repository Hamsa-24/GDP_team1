import numpy as np

def angle_entre_vecteurs(orientation_agent, position_agent, position_cible):
    # Vecteur qui va de l'agent à la cible
    vecteur_agent_cible = np.array(position_cible) - np.array(position_agent)
    
    # Convertir l'orientation de l'agent en vecteur unitaire
    orientation_unitaire = np.array([np.cos(orientation_agent), np.sin(orientation_agent)])
    
    # Calculer le produit scalaire entre les vecteurs orientation et agent-cible
    produit_scalaire = np.dot(orientation_unitaire, vecteur_agent_cible)
    
    # Calculer le produit vectoriel entre les vecteurs orientation et agent-cible
    produit_vectoriel = np.cross(orientation_unitaire, vecteur_agent_cible)
    
    # Calculer l'angle en radians
    angle_radians = np.arctan2(produit_vectoriel, produit_scalaire)
    
    # Convertir l'angle en degrés
    angle_degres = np.degrees(angle_radians)
    
    return angle_degres

# Exemple d'utilisation
orientation_agent = np.pi / 4  # Angle de l'agent en radians (exemple)
position_agent = [0, 0]  # Position de l'agent (exemple)
position_cible = [-6, -5]  # Position de la cible (exemple)

angle = angle_entre_vecteurs(orientation_agent, position_agent, position_cible)
print("Angle entre l'orientation de l'agent et le vecteur vers la cible:", angle, "degrés")
