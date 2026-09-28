from color_distance import calculate_distance
import numpy as np
import cv2

class ReactionAnalyzer:
    """
    Contains the Drug Reaction Dictionary.
    Analyzes the calibrated color of the test kit reaction and calculates 
    the Delta-E 2000 distance to known positive drug profiles.
    """

    # Named color lookup table (LAB space centroids for common color names)
    COLOR_NAMES = {
        "Black":        np.array([0.0, 0.0, 0.0]),
        "Dark Gray":    np.array([25.0, 0.0, 0.0]),
        "Gray":         np.array([50.0, 0.0, 0.0]),
        "Light Gray":   np.array([75.0, 0.0, 0.0]),
        "White":        np.array([100.0, 0.0, 0.0]),
        "Dark Red":     np.array([25.0, 45.0, 35.0]),
        "Red":          np.array([53.0, 80.0, 67.0]),
        "Orange":       np.array([70.0, 30.0, 55.0]),
        "Brown":        np.array([35.0, 20.0, 25.0]),
        "Dark Orange":  np.array([50.0, 35.0, 45.0]),
        "Yellow":       np.array([90.0, -5.0, 80.0]),
        "Olive":        np.array([50.0, -10.0, 40.0]),
        "Dark Green":   np.array([30.0, -40.0, 25.0]),
        "Green":        np.array([55.0, -55.0, 35.0]),
        "Teal":         np.array([50.0, -30.0, -15.0]),
        "Cyan":         np.array([70.0, -30.0, -30.0]),
        "Dark Blue":    np.array([20.0, 15.0, -50.0]),
        "Blue":         np.array([45.0, 25.0, -75.0]),
        "Purple":       np.array([35.0, 45.0, -50.0]),
        "Dark Purple":  np.array([15.0, 30.0, -40.0]),
        "Violet":       np.array([40.0, 55.0, -45.0]),
        "Magenta":      np.array([55.0, 75.0, -20.0]),
        "Pink":         np.array([70.0, 40.0, -5.0]),
        "Purple-Black": np.array([12.0, 15.0, -20.0]),
        "Blue-Green":   np.array([40.0, -25.0, -10.0]),
    }

    def __init__(self):
        # The Drug Dictionary
        # Defines the EXACT CIE LAB colors (under D65 illuminant) expected 
        # for a positive reaction on specific field test kits.
        # Kit IDs match the kits.py endpoint IDs.
        self.drug_profiles = {
            # Marquis: turns purple-black for MDMA, orange for amphetamine, red-purple for heroin
            "marquis": {
                "MDMA": np.array([15.0, 10.2, -15.5]),       # Dark purple/black
                "Amphetamine": np.array([45.0, 40.5, 30.2]), # Orange/brown
                "Heroin": np.array([25.0, 35.1, 5.0]),       # Purplish red
            },
            # Scott: turns blue for cocaine
            "scott": {
                "Cocaine": np.array([40.0, -10.0, -35.0]),   # Bright blue/turquoise
            },
            # Mandelin: turns orange for ketamine, dark blue for amphetamines
            "mandelin": {
                "Ketamine": np.array([30.0, 20.0, 35.0]),    # Deep orange/brown
                "Amphetamine": np.array([20.0, -5.0, -30.0]),# Dark blue
            },
            # Mecke: turns blue-green for opiates
            "mecke": {
                "Heroin": np.array([28.0, -20.0, -15.0]),    # Blue-green
                "Morphine": np.array([25.0, -18.0, -12.0]),  # Similar blue-green
                "Oxycodone": np.array([32.0, -15.0, -20.0]), # Blue
            },
            # Ehrlich: turns purple for LSD/indoles
            "ehrlich": {
                "LSD": np.array([35.0, 30.0, -40.0]),        # Vivid purple
                "DMT": np.array([30.0, 25.0, -35.0]),        # Purple
                "Psilocybin": np.array([28.0, 22.0, -38.0]), # Purple-violet
            },
            # Fallback for "other" or unknown kits — match against all known
            "other": {
                "Unknown Substance": np.array([50.0, 0.0, 0.0]),
            },
        }

        # Legacy name-based keys (so camera screen can map by kit name as fallback)
        self.drug_profiles["Marquis"] = self.drug_profiles["marquis"]
        self.drug_profiles["Scott"] = self.drug_profiles["scott"]
        self.drug_profiles["Mandelin"] = self.drug_profiles["mandelin"]

        # Delta-E thresholds
        # < 5.0 is a visually perfect match
        # 5.0 - 15.0 is a likely match (positive)
        # 15.0 - 25.0 is inconclusive (amber)
        # > 25.0 is a negative match
        self.POSITIVE_THRESHOLD = 15.0
        self.INCONCLUSIVE_THRESHOLD = 25.0

    @staticmethod
    def lab_to_rgb(lab_color):
        """Convert CIE LAB to RGB (0-255). Returns [R, G, B]."""
        # Convert to OpenCV LAB scale: L(0-255), a(0-255), b(0-255)
        l_cv = np.clip(lab_color[0] * 255.0 / 100.0, 0, 255)
        a_cv = np.clip(lab_color[1] + 128.0, 0, 255)
        b_cv = np.clip(lab_color[2] + 128.0, 0, 255)
        lab_pixel = np.uint8([[[int(l_cv), int(a_cv), int(b_cv)]]])
        rgb_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2RGB)
        return [int(rgb_pixel[0][0][0]), int(rgb_pixel[0][0][1]), int(rgb_pixel[0][0][2])]

    @staticmethod
    def rgb_to_hex(rgb):
        """Convert [R, G, B] to hex string like '#CC0000'."""
        r = max(0, min(255, int(rgb[0])))
        g = max(0, min(255, int(rgb[1])))
        b = max(0, min(255, int(rgb[2])))
        return f"#{r:02X}{g:02X}{b:02X}"

    @classmethod
    def get_color_name(cls, lab_color):
        """Find the closest named color to a LAB value."""
        best_name = "Unknown"
        best_dist = float('inf')
        lab_arr = np.array(lab_color)

        for name, ref_lab in cls.COLOR_NAMES.items():
            dist = float(np.sqrt(np.sum((lab_arr - ref_lab) ** 2)))
            if dist < best_dist:
                best_dist = dist
                best_name = name

        return best_name

    def analyze_reaction(self, kit_type: str, calibrated_reaction_lab):
        """
        Compares the calibrated reaction color against the known profiles for the selected kit.
        Returns the closest match, confidence score, hex code, color name, and status.
        kit_type can be the kit ID (e.g. 'marquis') or legacy name (e.g. 'Marquis').
        """
        # Try exact match first, then case-insensitive
        profiles = self.drug_profiles.get(kit_type)
        if profiles is None:
            profiles = self.drug_profiles.get(kit_type.lower())
        if profiles is None:
            # Try partial name matching
            for key in self.drug_profiles:
                if kit_type.lower() in key.lower() or key.lower() in kit_type.lower():
                    profiles = self.drug_profiles[key]
                    break

        if profiles is None:
            return {"status": "error", "message": f"Unknown test kit type: '{kit_type}'. Known types: {list(self.drug_profiles.keys())}"}

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
            confidence = 0.0  # High confidence it is negative

        # Convert calibrated LAB to RGB and hex for display
        corrected_rgb = self.lab_to_rgb(calibrated_reaction_lab)
        hex_code = self.rgb_to_hex(corrected_rgb)
        color_name = self.get_color_name(calibrated_reaction_lab)

        return {
            "result": result,
            "substance": best_match_name,
            "confidence": round(confidence, 1),
            "delta_e": round(best_match_distance, 2),
            "hex_code": hex_code,
            "color_name": color_name,
            "corrected_rgb": corrected_rgb,
        }

