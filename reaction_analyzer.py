from color_distance import calculate_distance
import numpy as np

class ReactionAnalyzer:
    """
    Contains the Drug Reaction Dictionary.
    Analyzes the calibrated color of the test kit reaction and calculates 
    the Delta-E 2000 distance to known positive drug profiles.
    """

    def __init__(self):
        # The Drug Dictionary
        # Defines the EXACT CIE LAB colors (under D65 illuminant) expected 
        # for a positive reaction on specific field test kits.
        self.drug_profiles = {
            "Marquis": {
                "MDMA": np.array([15.0, 10.2, -15.5]),      # Dark purple / Black
                "Amphetamine": np.array([45.0, 40.5, 30.2]),# Orange / Brown
                "Heroin": np.array([25.0, 35.1, 5.0])       # Purplish Red
            },
            "Scott": {
                "Cocaine": np.array([40.0, -10.0, -35.0])   # Bright Blue / Turquoise
            },
            "Mandelin": {
                "Ketamine": np.array([30.0, 20.0, 35.0])    # Deep Orange / Brown
            }
        }
        
        # Delta-E thresholds
        # < 5.0 is a visually perfect match
        # 5.0 - 15.0 is a likely match (positive)
        # 15.0 - 25.0 is inconclusive (amber)
        # > 25.0 is a negative match
        self.POSITIVE_THRESHOLD = 15.0
        self.INCONCLUSIVE_THRESHOLD = 25.0

    def analyze_reaction(self, kit_type, calibrated_reaction_lab):
        """
        Compares the calibrated reaction color against the known profiles for the selected kit.
        Returns the closest match, confidence score, and status.
        """
        if kit_type not in self.drug_profiles:
            return {"status": "error", "message": "Unknown test kit type."}

        profiles = self.drug_profiles[kit_type]
        best_match_name = None
        best_match_distance = float('inf')

        # Find the closest matching drug profile
        for drug_name, expected_lab in profiles.items():
            distance = calculate_distance(calibrated_reaction_lab, expected_lab)
            if distance < best_match_distance:
                best_match_distance = distance
                best_match_name = drug_name

        # Calculate Confidence Score (0% to 99.9%)
        # A distance of 0 is 100% confidence. A distance of 30 is 0% confidence.
        confidence = max(0.0, min(99.9, 100.0 - (best_match_distance / 30.0 * 100.0)))

        # Determine Final Result
        if best_match_distance <= self.POSITIVE_THRESHOLD:
            result = "POSITIVE"
        elif best_match_distance <= self.INCONCLUSIVE_THRESHOLD:
            result = "INCONCLUSIVE"
        else:
            result = "NEGATIVE"
            best_match_name = "None"
            confidence = 0.0 # High confidence it is negative

        return {
            "result": result,
            "substance": best_match_name,
            "confidence": round(confidence, 1),
            "delta_e": round(best_match_distance, 2)
        }
