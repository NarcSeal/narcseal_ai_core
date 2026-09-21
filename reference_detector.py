import cv2
import numpy as np

class ReferenceDetector:
    """
    Detects the physical Reference Color Card in a raw camera frame.
    Checks lighting, applies perspective warp to flatten the card, 
    and extracts the 6 calibration patches.
    """

    def __init__(self):
        # We expect a card with a specific aspect ratio, e.g., 3:2 (Credit Card sized)
        self.card_width = 600
        self.card_height = 400

    def check_lighting(self, frame):
        """
        Check if the image is too dark.
        Returns 'ok' or 'too_dark' to trigger the UI flashlight prompt.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < 40:
            return "too_dark"
        return "ok"

    def detect_and_warp_card(self, frame):
        """
        Finds the largest rectangular contour (the card) and applies 
        a perspective transform to flatten it.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Sort contours by area, keep the largest
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
        card_contour = None

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            # If our approximated contour has 4 points, we assume it's the card
            if len(approx) == 4:
                card_contour = approx
                break

        if card_contour is None:
            return None # Card not detected clearly

        # Order the 4 points (top-left, top-right, bottom-right, bottom-left)
        pts = card_contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # Destination points for the perspective warp
        dst = np.array([
            [0, 0],
            [self.card_width - 1, 0],
            [self.card_width - 1, self.card_height - 1],
            [0, self.card_height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(frame, M, (self.card_width, self.card_height))
        
        return warped

    def extract_calibration_patches(self, warped_card):
        """
        Assuming the card has 6 color patches in known grid positions,
        extracts the average RGB value for each patch.
        """
        if warped_card is None:
            return None

        # These coordinates are hardcoded based on the physical design of the NarcSeal Reference Card
        # Format: (x_center, y_center)
        patch_centers = [
            (100, 100), # Patch 1 (White)
            (300, 100), # Patch 2 (Black)
            (500, 100), # Patch 3 (Red)
            (100, 300), # Patch 4 (Green)
            (300, 300), # Patch 5 (Blue)
            (500, 300)  # Patch 6 (Yellow)
        ]

        patch_radius = 20
        extracted_colors = []

        for x, y in patch_centers:
            # Crop a small box around the center of the patch
            roi = warped_card[y-patch_radius:y+patch_radius, x-patch_radius:x+patch_radius]
            # Calculate the average color in this box (BGR)
            avg_color_per_row = np.average(roi, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            # Convert BGR (OpenCV default) to RGB
            rgb_color = [avg_color[2], avg_color[1], avg_color[0]]
            extracted_colors.append(rgb_color)

        return extracted_colors
