import math

def delta_e_2000(lab1, lab2):
    """
    Calculates the Delta-E 2000 color distance between two LAB colors.
    This provides a distance metric that is closely aligned with human visual perception.
    A Delta-E of < 1.0 is barely perceptible. A Delta-E < 5.0 is a close match.
    
    Input: lab1 [L1, a1, b1], lab2 [L2, a2, b2]
    Output: Delta-E float value.
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    
    # Delta L prime
    delta_L_prime = L2 - L1
    
    # Calculate C1 and C2
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    
    # Calculate average C
    C_bar = (C1 + C2) / 2.0
    
    # Calculate G
    G = 0.5 * (1 - math.sqrt((C_bar**7) / (C_bar**7 + 25**7)))
    
    # Calculate a1_prime and a2_prime
    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2
    
    # Calculate C1_prime and C2_prime
    C1_prime = math.sqrt(a1_prime**2 + b1**2)
    C2_prime = math.sqrt(a2_prime**2 + b2**2)
    
    # Calculate delta C prime
    delta_C_prime = C2_prime - C1_prime
    
    # Calculate average C prime
    C_bar_prime = (C1_prime + C2_prime) / 2.0
    
    # Calculate h1_prime and h2_prime
    h1_prime = math.degrees(math.atan2(b1, a1_prime))
    if h1_prime < 0: h1_prime += 360
    
    h2_prime = math.degrees(math.atan2(b2, a2_prime))
    if h2_prime < 0: h2_prime += 360
    
    # Calculate delta h prime
    delta_h_prime = 0.0
    if C1_prime * C2_prime != 0:
        if abs(h2_prime - h1_prime) <= 180:
            delta_h_prime = h2_prime - h1_prime
        elif h2_prime - h1_prime > 180:
            delta_h_prime = h2_prime - h1_prime - 360
        else:
            delta_h_prime = h2_prime - h1_prime + 360
            
    # Calculate delta H prime
    delta_H_prime = 2 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(delta_h_prime / 2.0))
    
    # Calculate average h prime
    h_bar_prime = 0.0
    if C1_prime * C2_prime != 0:
        if abs(h1_prime - h2_prime) <= 180:
            h_bar_prime = (h1_prime + h2_prime) / 2.0
        elif h1_prime + h2_prime < 360:
            h_bar_prime = (h1_prime + h2_prime + 360) / 2.0
        else:
            h_bar_prime = (h1_prime + h2_prime - 360) / 2.0
            
    # Calculate T
    T = 1 - 0.17 * math.cos(math.radians(h_bar_prime - 30)) \
          + 0.24 * math.cos(math.radians(2 * h_bar_prime)) \
          + 0.32 * math.cos(math.radians(3 * h_bar_prime + 6)) \
          - 0.20 * math.cos(math.radians(4 * h_bar_prime - 63))
          
    # Calculate delta theta
    delta_theta = 30 * math.exp(-((h_bar_prime - 275) / 25)**2)
    
    # Calculate R_C
    R_C = 2 * math.sqrt((C_bar_prime**7) / (C_bar_prime**7 + 25**7))
    
    # Calculate S_L, S_C, S_H
    S_L = 1 + (0.015 * (L_bar - 50)**2) / math.sqrt(20 + (L_bar - 50)**2)
    S_C = 1 + 0.045 * C_bar_prime
    S_H = 1 + 0.015 * C_bar_prime * T
    
    # Calculate R_T
    R_T = -math.sin(math.radians(2 * delta_theta)) * R_C
    
    # Calculate Delta E 2000
    de2000 = math.sqrt(
        (delta_L_prime / S_L)**2 +
        (delta_C_prime / S_C)**2 +
        (delta_H_prime / S_H)**2 +
        R_T * (delta_C_prime / S_C) * (delta_H_prime / S_H)
    )
    
    return de2000

# We need to define L_bar as well
def calculate_distance(lab1, lab2):
    global L_bar 
    L1 = lab1[0]
    L2 = lab2[0]
    L_bar = (L1 + L2) / 2.0
    return delta_e_2000(lab1, lab2)
