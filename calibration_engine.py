import numpy as np
import cv2

class CalibrationEngine:
    """
    ChromaLock Color Calibration Engine
    This class handles the D65 illuminant conversion to CIE LAB and computes
    the 3x3 Color Correction Matrix (CCM) using Least-Squares Regression.
    """

    def __init__(self):
        # Known true LAB values of the 6 reference patches on the physical card.
        # Computed from measured hex codes under standard D65 illuminant:
        #   White #FFFFFF, Black #0A0A0A, Red #CC0000,
        #   Green #009A44, Blue #2555F5, Yellow #FFC300
        # Patch order: White, Black, Red, Green, Blue, Yellow
        self.reference_lab_values = np.array([
            [100.0,    0.0,    0.0],   # White  (#FFFFFF)
            [  1.6,    0.0,    0.0],   # Black  (#0A0A0A)
            [ 41.2,   61.4,   52.2],   # Red    (#CC0000)
            [ 55.4,  -52.8,   35.6],   # Green  (#009A44)
            [ 41.2,   35.6,  -80.2],   # Blue   (#2555F5)
            [ 81.1,    6.0,   82.4],   # Yellow (#FFC300)
        ])

    def rgb_to_lab(self, rgb_color):
        """
        Converts an RGB color to CIE LAB using standard D65 illuminant.
        Using OpenCV for fast conversion.
        Input: [R, G, B] array in 0-255 range.
        Output: [L, a, b] array.
        """
        # OpenCV requires a 3D numpy array for color conversions
        rgb_pixel = np.uint8([[rgb_color]])
        lab_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2LAB)
        
        # OpenCV LAB ranges are L(0-255), a(0-255), b(0-255). 
        # Convert to standard CIE LAB ranges: L(0-100), a(-127 to 127), b(-127 to 127)
        l, a, b = lab_pixel[0][0]
        true_l = (l * 100.0) / 255.0
        true_a = a - 128.0
        true_b = b - 128.0
        
        return np.array([true_l, true_a, true_b])

    def compute_ccm(self, captured_rgb_patches):
        """
        Computes the 3x3 Color Correction Matrix (CCM) using Least-Squares Regression.
        Input: 6 captured RGB colors (averages from the 6 card patches in the image).
        Output: 3x3 Numpy Matrix.
        """
        if len(captured_rgb_patches) != 6:
            raise ValueError("Exactly 6 captured patches are required.")

        # Convert captured RGB to standard LAB
        captured_lab_values = np.array([self.rgb_to_lab(rgb) for rgb in captured_rgb_patches])

        # We want to find a 3x3 Matrix 'M' such that:
        # Captured_LAB * M = Reference_LAB
        # Using Least-Squares Regression: M = pseudoinverse(Captured_LAB) * Reference_LAB
        
        # Compute the pseudoinverse of the captured LAB values
        captured_pinv = np.linalg.pinv(captured_lab_values)
        
        # Multiply by the reference LAB values to get the CCM
        ccm = np.dot(captured_pinv, self.reference_lab_values)
        
        return ccm

    def apply_ccm(self, ccm, test_strip_rgb):
        """
        Applies the computed CCM to the raw RGB of the test strip reaction.
        Returns the corrected LAB color of the drug test reaction.
        """
        test_lab = self.rgb_to_lab(test_strip_rgb)
        corrected_lab = np.dot(test_lab, ccm)
        return corrected_lab

# Example Usage for Flutter Developer:
# engine = CalibrationEngine()
# captured = [[240,240,240], [10,10,10], [200,50,50], [50,200,50], [50,50,200], [200,200,50]]
# ccm = engine.compute_ccm(captured)
# raw_reaction = [150, 40, 200]
# corrected_reaction = engine.apply_ccm(ccm, raw_reaction)
