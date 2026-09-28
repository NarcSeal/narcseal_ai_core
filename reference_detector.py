import cv2
import numpy as np


class ReferenceDetector:
    """
    Detects the NarcSeal Reference Color Card anywhere in the camera frame.

    PHILOSOPHY:
        The card has 6 coloured patches on a white background.
        Detection does NOT match exact hex codes.
        It simply finds "a white rectangle with 6 distinct colours in a grid."
        Once found, the raw patch colours are extracted for calibration --
        the calibration engine uses the DIFFERENCE between expected and
        observed colours to compute a correction matrix.

    Card layout (physical):
        Row 0 (top):    White  | Black | Red
        Row 1 (bottom): Green  | Blue  | Yellow

    Pipeline:
      1. Lighting check
      2. Multi-strategy contour search (very loose geometry)
      3. Perspective warp to 600x400
      4. SOFT colour verification: just checks that there are multiple
         distinct colours in a 3x2 grid -- no exact HSV matching
      5. Return warped card for calibration
    """

    CARD_WIDTH  = 600
    CARD_HEIGHT = 400

    # Very loose geometry -- detect the card at any angle/distance
    MIN_ASPECT_RATIO = 1.0
    MAX_ASPECT_RATIO = 3.0
    MIN_AREA_FRACTION = 0.003   # 0.3% of frame
    MAX_AREA_FRACTION = 0.95

    MIN_BRIGHTNESS = 25
    MIN_SHARPNESS  = 5.0  # Very low -- let colour check be the real gate

    # Grid layout for patch sampling (in 600x400 warped space)
    # 3 columns x 2 rows
    PATCH_CENTERS = [
        (100, 110),   # Row 0, Col 0  (White)
        (300, 110),   # Row 0, Col 1  (Black)
        (500, 110),   # Row 0, Col 2  (Red)
        (100, 290),   # Row 1, Col 0  (Green)
        (300, 290),   # Row 1, Col 1  (Blue)
        (500, 290),   # Row 1, Col 2  (Yellow)
    ]
    PATCH_RADIUS = 45

    # For soft detection: minimum number of DISTINCT colours needed
    MIN_DISTINCT_COLOURS = 5      # out of 6 patches — require at least 5 distinct colors
    COLOUR_DISTANCE_THRESH = 25.0 # Euclidean distance in BGR to count as "distinct"

    # For soft detection: we also check for a very bright and very dark patch
    BRIGHT_THRESHOLD = 180  # V channel average for "bright" (white patch)
    DARK_THRESHOLD   = 80   # V channel average for "dark" (black patch)

    def __init__(self):
        self.card_width  = self.CARD_WIDTH
        self.card_height = self.CARD_HEIGHT

    # ==========================================================
    #  PUBLIC API
    # ==========================================================

    def check_lighting(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return "ok" if np.mean(gray) >= self.MIN_BRIGHTNESS else "too_dark"

    def detect_and_warp_card(self, frame):
        """
        Find the reference card ANYWHERE in the frame.
        Uses soft detection: looks for a rectangle with multiple distinct colours.
        Returns warped 600x400 image, or None.
        """
        frame_area = frame.shape[0] * frame.shape[1]
        candidates = self._find_rect_candidates(frame, frame_area)

        print(f"[Detector] {len(candidates)} candidates in {frame.shape[1]}x{frame.shape[0]}")

        for i, contour in enumerate(candidates):
            area = cv2.contourArea(contour)
            pct  = (area / frame_area) * 100
            print(f"  #{i}: area={area:.0f} ({pct:.1f}%)")

            warped = self._perspective_warp(frame, contour)
            if warped is None:
                print(f"  #{i}: warp failed")
                continue

            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            if sharpness < self.MIN_SHARPNESS:
                print(f"  #{i}: too blurry ({sharpness:.1f})")
                continue

            # Try all 4 orientations with SOFT colour check
            for label, img in [
                ("0deg",   warped),
                ("180deg", cv2.rotate(warped, cv2.ROTATE_180)),
                ("90cw",   cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)),
                ("90ccw",  cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)),
            ]:
                ok, detail = self._soft_verify(img)
                print(f"  #{i} ({label}): {detail}")
                if ok:
                    print(f"  #{i}: CARD DETECTED ({label})")
                    return img

        print("[Detector] No card found")
        return None

    def extract_calibration_patches(self, warped_card):
        """Extract average RGB for each of the 6 calibration patches."""
        if warped_card is None:
            return None
        extracted = []
        r = self.PATCH_RADIUS
        for x, y in self.PATCH_CENTERS:
            y0 = max(0, y - r)
            y1 = min(warped_card.shape[0], y + r)
            x0 = max(0, x - r)
            x1 = min(warped_card.shape[1], x + r)
            roi = warped_card[y0:y1, x0:x1]
            if roi.size == 0:
                extracted.append([128, 128, 128])
                continue
            avg = np.mean(roi.reshape(-1, 3), axis=0)
            extracted.append([float(avg[2]), float(avg[1]), float(avg[0])])  # BGR->RGB
        return extracted

    # ==========================================================
    #  SOFT VERIFICATION -- the key change
    # ==========================================================

    def _soft_verify(self, warped):
        """
        Strict colour verification against KNOWN reference card colors.
        
        Checks that the 6 sampled patches match the expected colors:
          Patch 0 (100,110): White   — high V, low S
          Patch 1 (500,110): Black   — low V
          Patch 2 (100,290): Red     — H near 0 or 170-180, high S
          Patch 3 (300,110): Green   — H near 35-85, high S
          Patch 4 (300,290): Blue    — H near 100-130, high S
          Patch 5 (500,290): Yellow  — H near 20-35, high S, high V
        
        Requires at least 5 of 6 patches to match their expected color.
        This prevents false positives from random objects like phones, 
        tables, plastic containers etc.
        """
        h, w = warped.shape[:2]
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        r = self.PATCH_RADIUS

        # Sample the 6 patch colours in HSV and BGR
        patches_hsv = []
        patches_bgr = []

        for x, y in self.PATCH_CENTERS:
            y0 = max(0, y - r)
            y1 = min(h, y + r)
            x0 = max(0, x - r)
            x1 = min(w, x + r)

            roi_bgr = warped[y0:y1, x0:x1]
            roi_hsv = hsv[y0:y1, x0:x1]

            if roi_bgr.size == 0:
                patches_hsv.append((0.0, 0.0, 128.0))
                patches_bgr.append(np.array([128.0, 128.0, 128.0]))
                continue

            avg_h = float(np.mean(roi_hsv[:, :, 0].astype(float)))
            avg_s = float(np.mean(roi_hsv[:, :, 1].astype(float)))
            avg_v = float(np.mean(roi_hsv[:, :, 2].astype(float)))
            avg_bgr = np.mean(roi_bgr.reshape(-1, 3).astype(float), axis=0)
            patches_hsv.append((avg_h, avg_s, avg_v))
            patches_bgr.append(avg_bgr)

        # Define color matchers for each patch position
        # Each returns True if the patch matches the expected color
        # Using generous but meaningful thresholds
        def is_white(h, s, v, bgr):
            # White: very low saturation, high value
            # Also check that BGR channels are all relatively high and close to each other
            b, g, r_ch = bgr
            channel_spread = max(b, g, r_ch) - min(b, g, r_ch)
            return s < 80 and v > 140 and channel_spread < 80

        def is_black(h, s, v, bgr):
            # Black: very low value
            return v < 90

        def is_red(h, s, v, bgr):
            # Red: hue near 0 (wraps around 170-180 in OpenCV) and high saturation
            # Also BGR: R channel should dominate
            b, g, r_ch = bgr
            hue_ok = (h < 15 or h > 160) and s > 60
            bgr_ok = r_ch > 80 and r_ch > b * 1.3 and r_ch > g * 1.3
            return hue_ok or bgr_ok

        def is_green(h, s, v, bgr):
            # Green: hue in 35-85 range, significant saturation
            b, g, r_ch = bgr
            hue_ok = 30 < h < 90 and s > 50 and v > 40
            bgr_ok = g > 60 and g > r_ch * 1.2 and g > b * 1.0
            return hue_ok or (bgr_ok and s > 40)

        def is_blue(h, s, v, bgr):
            # Blue: hue in 90-135 range, significant saturation
            b, g, r_ch = bgr
            hue_ok = 85 < h < 140 and s > 50 and v > 30
            bgr_ok = b > 80 and b > r_ch * 1.2 and b > g * 1.0
            return hue_ok or (bgr_ok and s > 40)

        def is_yellow(h, s, v, bgr):
            # Yellow: hue in 18-38 range, high saturation, high value
            b, g, r_ch = bgr
            hue_ok = 15 < h < 40 and s > 60 and v > 100
            bgr_ok = r_ch > 120 and g > 100 and b < r_ch * 0.7
            return hue_ok or bgr_ok

        # Check each patch against its expected color
        # Patch order: White(0), Black(1), Red(2), Green(3), Blue(4), Yellow(5)
        matchers = [is_white, is_black, is_red, is_green, is_blue, is_yellow]
        color_names = ["White", "Black", "Red", "Green", "Blue", "Yellow"]
        
        matches = 0
        match_details = []
        
        for i, (matcher, name) in enumerate(zip(matchers, color_names)):
            h_val, s_val, v_val = patches_hsv[i]
            bgr = patches_bgr[i]
            matched = matcher(h_val, s_val, v_val, bgr)
            matches += 1 if matched else 0
            status = "PASS" if matched else "FAIL"
            match_details.append(f"{name}:{status}(H={h_val:.0f},S={s_val:.0f},V={v_val:.0f})")

        detail = f"matches={matches}/6 [{', '.join(match_details)}]"
        
        # Require at least 5 of 6 patches to match
        passed = matches >= self.MIN_DISTINCT_COLOURS
        
        with open("../narcseal_backend/detection_debug.log", "a") as f:
            f.write(f"  soft_verify: passed={passed}, {detail}\n")
            
        return passed, detail

    # ==========================================================
    #  CONTOUR DETECTION
    # ==========================================================

    def _find_rect_candidates(self, frame, frame_area):
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        candidates = []

        # Strategy 1: Canny with multiple thresholds
        for lo, hi in [(20, 80), (30, 100), (50, 150), (70, 200), (100, 250)]:
            edged = cv2.Canny(blurred, lo, hi)
            edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=2)
            candidates += self._extract_quads(edged, frame_area)

        # Strategy 2: Adaptive threshold
        for bs in [11, 21, 31]:
            adaptive = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, bs, 2,
            )
            candidates += self._extract_quads(adaptive, frame_area)

        # Strategy 3: OTSU
        _, otsu     = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)
        _, otsu_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        candidates += self._extract_quads(otsu, frame_area)
        candidates += self._extract_quads(otsu_inv, frame_area)

        # Strategy 4: White region detection (card has white background)
        _, white = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=3)
        candidates += self._extract_quads(white, frame_area)

        # Strategy 5: Lower white threshold for darker lighting
        _, white2 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        white2 = cv2.morphologyEx(white2, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=3)
        candidates += self._extract_quads(white2, frame_area)

        # Strategy 6: Colour blob detection -- find any saturated regions
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat_mask = cv2.inRange(hsv, (0, 50, 40), (180, 255, 255))
        sat_mask = cv2.dilate(sat_mask, np.ones((25, 25), np.uint8), iterations=3)
        candidates += self._extract_quads(sat_mask, frame_area)

        # Deduplicate and sort
        candidates = self._deduplicate(candidates)
        candidates.sort(key=cv2.contourArea, reverse=True)
        return candidates[:20]

    def _extract_quads(self, mask, frame_area):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        quads = []
        seen  = set()

        for c in contours:
            area = cv2.contourArea(c)
            if area < frame_area * self.MIN_AREA_FRACTION:
                continue
            if area > frame_area * self.MAX_AREA_FRACTION:
                continue

            area_key = int(area / 300)
            if area_key in seen:
                continue

            peri  = cv2.arcLength(c, True)

            # Method A: approxPolyDP
            found = False
            for eps in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
                approx = cv2.approxPolyDP(c, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    if self._check_aspect(approx):
                        quads.append(approx)
                        seen.add(area_key)
                        found = True
                        break

            if found:
                continue

            # Method B: minAreaRect
            rect = cv2.minAreaRect(c)
            (_, _), (w, h), _ = rect
            if w == 0 or h == 0:
                continue
            ar = max(w, h) / min(w, h)
            if ar < self.MIN_ASPECT_RATIO or ar > self.MAX_ASPECT_RATIO:
                continue
            # Very loose solidity
            if area / (w * h) < 0.40:
                continue

            box = cv2.boxPoints(rect).astype(np.intp)
            quads.append(box.reshape(-1, 1, 2))
            seen.add(area_key)

        return quads

    def _check_aspect(self, approx):
        """Check if a 4-point contour has acceptable aspect ratio."""
        pts = approx.reshape(4, 2).astype(float)
        sides = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
        w_est = (sides[0] + sides[2]) / 2
        h_est = (sides[1] + sides[3]) / 2
        if w_est == 0 or h_est == 0:
            return False
        ar = max(w_est, h_est) / min(w_est, h_est)
        return self.MIN_ASPECT_RATIO <= ar <= self.MAX_ASPECT_RATIO

    @staticmethod
    def _deduplicate(candidates, dist_thresh=50.0):
        if not candidates:
            return []
        unique = [candidates[0]]
        for c in candidates[1:]:
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            is_dup = False
            for u in unique:
                Mu = cv2.moments(u)
                if Mu["m00"] == 0:
                    continue
                ux = Mu["m10"] / Mu["m00"]
                uy = Mu["m01"] / Mu["m00"]
                if np.hypot(cx - ux, cy - uy) < dist_thresh:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(c)
        return unique

    # -- Perspective warp

    def _perspective_warp(self, frame, contour):
        try:
            pts  = contour.reshape(4, 2).astype("float32")
            rect = self._order_points(pts)
            dst  = np.array([
                [0,                   0],
                [self.card_width - 1, 0],
                [self.card_width - 1, self.card_height - 1],
                [0,                   self.card_height - 1],
            ], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            return cv2.warpPerspective(frame, M, (self.card_width, self.card_height))
        except Exception as e:
            print(f"  Warp error: {e}")
            return None

    @staticmethod
    def _order_points(pts):
        rect    = np.zeros((4, 2), dtype="float32")
        s       = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        d       = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]
        return rect

    # -- Drug kit detection (outside the card area in the full frame)

    def detect_drug_kit(self, frame, card_contour):
        """
        Detect a drug test kit / reaction vessel in the full frame, OUTSIDE the reference card.
        
        Strategy:
        1. Mask out the reference card region from the frame
        2. Look for a significant colored region (saturated, distinct from background)
        3. Return the region's bounding box and average color, or None if not found
        """
        if frame is None or card_contour is None:
            return None

        h, w = frame.shape[:2]
        frame_area = h * w

        # Create mask of the card area (to exclude it)
        card_mask = np.zeros((h, w), dtype=np.uint8)
        # Expand the card contour slightly to fully exclude card + edges
        try:
            pts = card_contour.reshape(-1, 2).astype(np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(card_mask, hull, 255)
            # Dilate to add margin around card
            card_mask = cv2.dilate(card_mask, np.ones((31, 31), np.uint8), iterations=2)
        except Exception:
            return None

        # Invert mask — we want everything OUTSIDE the card
        outside_mask = cv2.bitwise_not(card_mask)

        # Look for colored/saturated regions outside the card
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Strategy 1: Find saturated regions (drug reactions are usually colored)
        # Sat > 30, Val > 40 to avoid shadows
        sat_mask = cv2.inRange(hsv, (0, 30, 40), (180, 255, 255))
        sat_mask = cv2.bitwise_and(sat_mask, outside_mask)
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=1)

        # Strategy 2: Also try finding any distinct region (even unsaturated like white/gray reaction)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Use edge detection to find distinct objects
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_and(edges, outside_mask)
        edges = cv2.dilate(edges, np.ones((11, 11), np.uint8), iterations=3)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8), iterations=2)

        # Combine both masks
        combined_mask = cv2.bitwise_or(sat_mask, edges)

        # Find contours in the combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter: drug kit should be a reasonably sized object
        min_kit_area = frame_area * 0.005   # At least 0.5% of frame
        max_kit_area = frame_area * 0.40    # No more than 40% of frame

        best_candidate = None
        best_score = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_kit_area or area > max_kit_area:
                continue

            x, y, bw, bh = cv2.boundingRect(c)

            # Extract the region
            roi = frame[y:y+bh, x:x+bw]
            if roi.size == 0:
                continue

            # Score: prefer regions with color (saturation) and reasonable size
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            avg_sat = float(np.mean(roi_hsv[:, :, 1]))
            avg_val = float(np.mean(roi_hsv[:, :, 2]))
            color_std = float(np.std(roi.reshape(-1, 3).astype(float)))

            score = (avg_sat * 0.4) + (color_std * 0.3) + (min(area / min_kit_area, 10) * 3.0)

            if score > best_score:
                best_score = score
                best_candidate = {
                    'bbox': (x, y, bw, bh),
                    'area': area,
                    'avg_saturation': avg_sat,
                    'avg_value': avg_val,
                    'color_std': color_std,
                    'score': score,
                }

        if best_candidate is None or best_score < 25.0:
            print(f"[KitDetect] No drug kit region found outside card area (best_score={best_score:.1f})")
            return None

        print(f"[KitDetect] Found kit region: bbox={best_candidate['bbox']} "
              f"area={best_candidate['area']:.0f} sat={best_candidate['avg_saturation']:.1f} "
              f"score={best_candidate['score']:.1f}")
        return best_candidate

    def extract_kit_reaction_color(self, frame, kit_info):
        """
        Extract the average reaction color from the detected drug kit region.
        Returns [R, G, B] in 0-255 range, or None.
        """
        if frame is None or kit_info is None:
            return None

        x, y, bw, bh = kit_info['bbox']

        # Sample the center 60% of the region to avoid edges
        margin_x = int(bw * 0.2)
        margin_y = int(bh * 0.2)
        center_roi = frame[
            y + margin_y : y + bh - margin_y,
            x + margin_x : x + bw - margin_x
        ]

        if center_roi.size == 0:
            # Fallback to full region
            center_roi = frame[y:y+bh, x:x+bw]

        if center_roi.size == 0:
            return None

        avg_bgr = np.mean(center_roi.reshape(-1, 3).astype(float), axis=0)
        # BGR → RGB
        return [float(avg_bgr[2]), float(avg_bgr[1]), float(avg_bgr[0])]

    def get_card_contour(self, frame):
        """
        Find the reference card contour (without warping).
        Returns the contour of the detected card, or None.
        Used to determine the card's location for kit detection.
        """
        frame_area = frame.shape[0] * frame.shape[1]
        candidates = self._find_rect_candidates(frame, frame_area)

        for contour in candidates:
            warped = self._perspective_warp(frame, contour)
            if warped is None:
                continue

            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            if sharpness < self.MIN_SHARPNESS:
                continue

            for label, img in [
                ("0deg",   warped),
                ("180deg", cv2.rotate(warped, cv2.ROTATE_180)),
                ("90cw",   cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)),
                ("90ccw",  cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)),
            ]:
                ok, detail = self._soft_verify(img)
                if ok:
                    return contour

        return None
