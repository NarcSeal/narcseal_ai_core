import cv2
import numpy as np
from reference_detector import ReferenceDetector
from calibration_engine import CalibrationEngine
from reaction_analyzer import ReactionAnalyzer

class ChromaLockEngine:
    """
    The Master Orchestrator for the NarcSeal AI.
    The Flutter mobile app should call 'process_frame' on every camera tick
    during the 3-second 'Valid Window'.
    """

    def __init__(self):
        self.detector = ReferenceDetector()
        self.calibrator = CalibrationEngine()
        self.analyzer = ReactionAnalyzer()

    def process_frame(self, frame_path, kit_type):
        """
        Main pipeline.
        1. Reads image.
        2. Checks lighting.
        3. Detects reference card.
        4. Extracts patches & computes CCM.
        5. Isolates test strip and applies CCM.
        6. Analyzes drug reaction and returns JSON.
        """
        # 1. Load image (simulating camera frame)
        frame = cv2.imread(frame_path)
        if frame is None:
            return {"status": "error", "message": "Failed to read image."}

        # 2. Check Lighting
        lighting = self.detector.check_lighting(frame)
        if lighting == "too_dark":
            return {"status": "error", "action": "enable_flashlight", "message": "Lighting is too dark."}

        # 3. Detect and Warp Card
        warped_card = self.detector.detect_and_warp_card(frame)
        if warped_card is None:
            return {"status": "scanning", "message": "Searching for Reference Card..."}

        # 4. Extract Calibration Patches
        captured_rgb_patches = self.detector.extract_calibration_patches(warped_card)
        
        # 5. Compute Color Correction Matrix (ChromaLock)
        try:
            ccm = self.calibrator.compute_ccm(captured_rgb_patches)
        except Exception as e:
            return {"status": "error", "message": "Failed to calibrate ChromaLock."}

        # 6. Extract Test Strip Reaction (Simulated coordinates)
        # In the real physical card, the test kit is placed in a cutout in the center
        test_strip_roi = warped_card[150:250, 250:350]
        avg_bgr = np.average(np.average(test_strip_roi, axis=0), axis=0)
        raw_reaction_rgb = [avg_bgr[2], avg_bgr[1], avg_bgr[0]]

        # Apply the CCM to get the true, lighting-independent LAB color
        calibrated_lab = self.calibrator.apply_ccm(ccm, raw_reaction_rgb)

        # 7. Analyze Reaction
        analysis_result = self.analyzer.analyze_reaction(kit_type, calibrated_lab)
        
        # Combine into final payload for UI
        final_response = {
            "status": "success",
            "lighting_check": "ok",
            "card_detected": True,
            "calibration": "complete",
            "result": analysis_result["result"],
            "substance": analysis_result["substance"],
            "confidence": analysis_result["confidence"],
            "explainable_delta_e": analysis_result["delta_e"]
        }
        
        return final_response

# Example Usage
# engine = ChromaLockEngine()
# result_json = engine.process_frame("test_photo.jpg", "Scott")
# print(result_json)
