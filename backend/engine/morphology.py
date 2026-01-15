import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import os
import io
from typing import Dict, Any
from PIL import Image, ImageOps

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

LANDMARKS = {
    # Vertical Midline
    "trichion": 10, "glabella": 168, "nasion": 6, "noseTip": 1, "subnasale": 164, 
    "lipTop": 0, "lipUpperBottom": 13, "lipLowerTop": 14, "menton": 152,
    
    # Brows
    "browLeftInner": 107, "browLeftOuter": 70, 
    "browRightInner": 336, "browRightOuter": 300,

    # Eyes
    "eyeLeftInner": 133, "eyeLeftOuter": 33,   
    "eyeLeftTop": 159, "eyeLeftBottom": 145,
    "eyeLeftTopInner": 160, "eyeLeftTopOuter": 158,
    "eyeLeftBottomInner": 144, "eyeLeftBottomOuter": 153,
    
    "eyeRightInner": 362, "eyeRightOuter": 263, 
    "eyeRightTop": 386, "eyeRightBottom": 374,
    "eyeRightTopInner": 385, "eyeRightTopOuter": 387,
    "eyeRightBottomInner": 380, "eyeRightBottomOuter": 373,
    
    # Cheek / Jaw
    "zygomaLeft": 234, "zygomaRight": 454,
    "gonionLeft": 58, "gonionRight": 288,
    "chinLeft": 172, "chinRight": 397,
    
    # Nose/Mouth
    "noseAlareLeft": 129, "noseAlareRight": 358,
    "mouthLeft": 61, "mouthRight": 291,
    "lipTop": 0, "lipBottom": 17, # Basic height
    "lipUpperInner": 13, "lipLowerInner": 14, # Inner height

    # Ears (Approx)
    "earLeft": 234, "earRight": 454 
}

class MorphologyEngine:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), '../face_landmarker.task')
        if not os.path.exists(model_path): print(f"WARNING: Model not found at {model_path}")

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            output_face_blendshapes=True
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) and Gamma Correction
        to normalize lighting and improve landmark detection accuracy.
        """
        # 1. CLAHE in LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        processed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # 2. Gamma Correction (Normalize brightness)
        gamma = 1.2
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(processed, table)

    def process_image(self, image_bytes: bytes) -> Dict[str, Any]:
        # 1. Load with PIL to handle EXIF rotation automatically
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            pil_image = ImageOps.exif_transpose(pil_image)
            
            # Convert to RGB (PIL) -> BGR (OpenCV)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
                
            image = np.array(pil_image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            # Fallback for non-image data or failures
            print(f"PIL Load Failed: {e}, falling back to cv2 raw decode")
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not decode image")
            
        # Apply pre-processing for better landmark mapping
        processed_image = self._preprocess_image(image)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        detection_result = self.landmarker.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return None
            
        landmarks = detection_result.face_landmarks[0]
        h, w, _ = image.shape
        points = {k: np.array([landmarks[v].x * w, landmarks[v].y * h]) for k, v in LANDMARKS.items()}
        
        metrics = self._calculate_metrics(points)
        
        # Add Normalized Landmarks
        lm_list = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks]
        
        # --- EXTRAPOLATE FOREHEAD (Refind Grid for Natural Coverage) ---
        # Vector from Glabella (168) to Trichion (10)
        try:
            g = landmarks[168]
            t = landmarks[10]
            # Direction vector (upwards)
            vx = t.x - g.x
            vy = t.y - g.y
            
            # Face width reference for horizontal spacing (use cheekbones)
            face_w = abs(landmarks[454].x - landmarks[234].x)
            step_x = face_w * 0.11 # Wide spacing
            
            # Generate a compact forehead strip (2 rows only - per user "2 dots too high")
            for row in range(1, 3):
                v_scale = 0.28 * row # Compact height
                # Minimal taper
                width_factor = 1.0 - (row * 0.05) 
                
                # Center point
                base_x = t.x + vx * v_scale
                base_y = t.y + vy * v_scale
                lm_list.append({"x": base_x, "y": base_y, "z": t.z})
                
                # Side points (fan out with curvature)
                for s in range(1, 6):
                    h_offset = (step_x * s) * width_factor
                    # Curvature: slight drop
                    curve_drop = (s * 0.015) * face_w
                    
                    # Left
                    lm_list.append({"x": base_x - h_offset, "y": base_y + curve_drop, "z": t.z})
                    # Right
                    lm_list.append({"x": base_x + h_offset, "y": base_y + curve_drop, "z": t.z})
                    
        except Exception as e:
            print(f"Forehead extrapolation error: {e}") 

        metrics["landmarks"] = lm_list
        return metrics

    def _calculate_metrics(self, p: Dict[str, np.ndarray]) -> Dict[str, Any]:
        def d(k1, k2): return np.linalg.norm(p[k1] - p[k2])
        
        # --- 1. BASIC DIMENSIONS ---
        bizygoma = d("zygomaLeft", "zygomaRight")
        face_h = d("trichion", "menton")
        bigonial = d("gonionLeft", "gonionRight")
        phi_ratio = face_h / bizygoma if bizygoma > 0 else 1.6
        fwhr = bizygoma / d("glabella", "lipTop") if d("glabella", "lipTop") > 0 else 1.9

        # --- 2. VERTICAL THIRDS ---
        u_h = d("trichion", "glabella")
        m_h = d("glabella", "subnasale")
        l_h = d("subnasale", "menton")
        tot_h = u_h + m_h + l_h
        thirds = {"upper": u_h/tot_h*100, "mid": m_h/tot_h*100, "lower": l_h/tot_h*100}
        
        lower_upper = d("subnasale", "lipTop")
        lower_lower = d("lipBottom", "menton") 
        lower_ratio = lower_lower / lower_upper if lower_upper > 0 else 2.0
        
        # Chin-Philtrum Ratio (same as lower ratio)
        chin_philtrum_ratio = lower_ratio
        
        # --- 3. HORIZONTAL RULE OF FIFTHS ---
        w_face_full = d("earLeft", "earRight")
        w_zygoma = bizygoma
        w_eye_l = d("eyeLeftInner", "eyeLeftOuter")
        w_eye_r = d("eyeRightInner", "eyeRightOuter")
        w_inters = d("eyeLeftInner", "eyeRightInner")
        
        # ESR: Inter-eye vs Avg Eye Width
        avg_eye_w = (w_eye_l + w_eye_r) / 2
        esr = w_inters / avg_eye_w if avg_eye_w > 0 else 1.0
        
        expected_face_w = avg_eye_w * 5
        face_width_deviation = abs(w_face_full - expected_face_w) / expected_face_w
        fifths_score = 100 * max(0, 1 - (face_width_deviation * 2)) # Strict

        # --- 4. EYES & GAZE ---
        tilt_l = (p["eyeLeftInner"][1] - p["eyeLeftOuter"][1]) / d("eyeLeftInner", "eyeLeftOuter") * 100
        tilt_r = (p["eyeRightInner"][1] - p["eyeRightOuter"][1]) / d("eyeRightInner", "eyeRightOuter") * 100
        avg_eye_tilt = (tilt_l + tilt_r) / 2
        
        eye_area_l = self._poly_area([p[k] for k in ["eyeLeftInner", "eyeLeftTop", "eyeLeftTopOuter", "eyeLeftOuter", "eyeLeftBottomOuter", "eyeLeftBottom"]])
        eye_area_r = self._poly_area([p[k] for k in ["eyeRightInner", "eyeRightTop", "eyeRightTopOuter", "eyeRightOuter", "eyeRightBottomOuter", "eyeRightBottom"]])
        eye_area_ratio = (eye_area_l + eye_area_r) / (face_h * w_face_full) * 100

        # --- 5. GOLDEN RATIOS ---
        nose_w = d("noseAlareLeft", "noseAlareRight")
        mouth_w = d("mouthLeft", "mouthRight")
        mouth_nose_ratio = mouth_w / nose_w if nose_w > 0 else 1.618

        # --- 6. SCORING (PEAK-REWARD SYSTEM) ---
        def peak_reward_score(x, target, sigma_low, sigma_high, cap=100):
            """
            Asymmetric scoring that rewards values approaching 'target':
            - Linear growth as x approaches target from below
            - Gentle Gaussian falloff for extreme values beyond target
            - Allows exceptional traits to score very high
            """
            if x <= target:
                # Below target: linear growth (80-100 range for reasonable values)
                normalized = (x / target) if target > 0 else 0
                score = 80 + (20 * normalized)
            else:
                # Above target: gentle falloff (stays high for moderate excess)
                excess = (x - target) / sigma_high
                score = 100 * np.exp(-0.5 * excess ** 2)
            
            return max(0, min(cap, score))
        
        def strict_gaussian(x, min_v, max_v, sigma):
            """Traditional gaussian for strict metrics like symmetry"""
            if min_v <= x <= max_v:
                return 100.0
            target = min_v if x < min_v else max_v
            score = 100 * np.exp(-0.5 * ((x - target) / sigma) ** 2)
            return max(0, min(100, score))

        # A. SYMMETRY (Very robust to real-world photo conditions)
        midline_x = (p["glabella"][0] + p["menton"][0]) / 2
        def get_sym(kL, kR): 
            asymmetry_pct = (abs(abs(p[kL][0]-midline_x) - abs(p[kR][0]-midline_x)) / bizygoma) * 100
            # VERY forgiving: Each 1% asymmetry costs only 5 points
            # This allows for head tilt, slight rotation,  lighting, etc.
            score = 100 - (asymmetry_pct * 5) 
            return max(50, min(100, score)) # Floor at 50, not 20
        
        sym_eyes = get_sym("eyeLeftOuter", "eyeRightOuter")
        sym_jaw = get_sym("gonionLeft", "gonionRight")
        sym_nose = get_sym("noseAlareLeft", "noseAlareRight")
        s_symmetry = (sym_eyes + sym_jaw + sym_nose) / 3

        # B. PROPORTIONS
        s_thirds = strict_gaussian(abs(thirds["upper"]-thirds["lower"]), 0, 15, 12)
        
        # CHIN-PHILTRUM (Fixed calculation using more reliable vertical measurements)
        # The issue: lipBottom-menton often gives too high values
        # Better approach: Use subnasale as top reference
        
        # Method 1: Traditional (philtrum vs chin)
        phil_h_traditional = d("subnasale", "lipTop")
        chin_h_traditional = d("lipBottom", "menton")
        
        # Method 2: Use lower third proportions (more stable)
        lower_third_h = d("subnasale", "menton")  
        # Typical split: philtrum ~40%, lips ~10%, chin ~50%
        phil_h_estimate = lower_third_h * 0.40
        chin_h_estimate = lower_third_h * 0.50
        
        # Use the method that gives more reasonable values (2.0-3.0 range)
        ratio_traditional = chin_h_traditional / phil_h_traditional if phil_h_traditional > 0 else 2.2
        ratio_estimate = chin_h_estimate / phil_h_estimate if phil_h_estimate > 0 else 2.2
        
        # Blend: if traditional is extreme, use estimate
        if 1.5 <= ratio_traditional <= 3.5:
            chin_philtrum_ratio = ratio_traditional
        else:
            chin_philtrum_ratio = ratio_estimate
            
        # Clamp to reasonable range
        chin_philtrum_ratio = max(1.5, min(3.5, chin_philtrum_ratio))
        lower_ratio = chin_philtrum_ratio  # backward compat
        
        s_low_ratio = strict_gaussian(chin_philtrum_ratio, 1.8, 2.8, 0.6)  # More forgiving range
        s_fifths_match = fifths_score

        # C. GOLDEN RATIOS
        s_phi = strict_gaussian(phi_ratio, 1.4, 1.8, 0.4)
        s_mouth_nose = strict_gaussian(mouth_nose_ratio, 1.2, 2.0, 0.6)

        # D. SEXUAL DIMORPHISM (PEAK REWARD - Higher is better!)
        # Jaw width ratio: Reward wide jaws aggressively
        jaw_ratio = bigonial/bizygoma
        if jaw_ratio >= 0.88:
            s_jaw = min(100, 85 + ((jaw_ratio - 0.88) * 80))  # 0.88->85, 0.95->90, 1.1->100
        else:
            s_jaw = 60 + ((jaw_ratio / 0.88) * 25) # 0.7->80
        
        # Canthal Tilt: Reward positive tilt aggressively
        if avg_eye_tilt >= 3:
            s_tilt = min(100, 85 + (avg_eye_tilt * 2))  # 3°->91, 7°->100
        elif avg_eye_tilt >= 0:
            s_tilt = 75 + (avg_eye_tilt * 3) 
        else:
            s_tilt = max(40, 75 + (avg_eye_tilt * 8))

        # --- 7. TRAIT SYNERGIES (Multiplicative Bonuses) ---
        # Elite combination: High symmetry + Strong jaw + Positive tilt
        has_elite_combo = (s_symmetry > 75 and s_jaw > 88 and s_tilt > 88)  # Lowered symmetry threshold
        synergy_bonus = 1.12 if has_elite_combo else 1.0 # 12% bonus for combo
        
        # Symmetry bonus but more attainable
        symmetry_bonus = 1.0
        if s_symmetry > 85:
            symmetry_bonus = 1.05
        elif s_symmetry > 75:
            symmetry_bonus = 1.02

        # --- 8. FINAL AGGREGATION (Reduced symmetry weight since it's unreliable) ---
        base_score = (
            0.15 * s_symmetry +       # Reduced from 0.20 (too punishing for photos)
            0.25 * s_jaw +            # Jaw is huge for attractiveness
            0.15 * s_fifths_match +   
            0.18 * s_tilt +           # Tilt is definitive
            0.10 * s_phi +            
            0.08 * s_thirds +         
            0.05 * s_mouth_nose +     
            0.04 * s_low_ratio        # Reduced since calculation is unstable
        )
        
        # Apply bonuses
        overall_score = base_score * synergy_bonus * symmetry_bonus
        
        # Clamp to 0-100
        overall_score = max(0, min(100, overall_score))

        # === DEBUG OUTPUT ===
        print(f"\n{'='*60}")
        print(f"DEBUG: Facial Analysis Breakdown")
        print(f"{'='*60}")
        print(f"RAW MEASUREMENTS:")
        print(f"  Jaw Ratio (bigonial/bizygoma): {bigonial/bizygoma:.3f}")
        print(f"  Canthal Tilt: {avg_eye_tilt:.2f}°")
        print(f"  Chin-Philtrum Ratio: {chin_philtrum_ratio:.3f}")
        print(f"  Lower Ratio: {lower_ratio:.3f}")
        print(f"  Phi Ratio: {phi_ratio:.3f}")
        print(f"  Mouth-Nose Ratio: {mouth_nose_ratio:.3f}")
        print(f"\nINDIVIDUAL SCORES:")
        print(f"  s_symmetry: {s_symmetry:.1f}")
        print(f"  s_jaw: {s_jaw:.1f}")
        print(f"  s_tilt: {s_tilt:.1f}")
        print(f"  s_fifths_match: {s_fifths_match:.1f}")
        print(f"  s_phi: {s_phi:.1f}")
        print(f"  s_thirds: {s_thirds:.1f}")
        print(f"  s_mouth_nose: {s_mouth_nose:.1f}")
        print(f"  s_low_ratio: {s_low_ratio:.1f}")
        print(f"\nAGGREGATION:")
        print(f"  Base Score: {base_score:.2f}")
        print(f"  Synergy Bonus: {synergy_bonus:.2f}x")
        print(f"  Symmetry Bonus: {symmetry_bonus:.2f}x")
        print(f"  FINAL SCORE: {overall_score:.1f}")
        print(f"{'='*60}\n")

        verdict = []
        if overall_score > 90: verdict.append("Elite Model Tier Aesthetics.")
        elif overall_score > 80: verdict.append("High Aesthetic Harmony.")
        
        if s_fifths_match > 90: verdict.append("Ideal horizontal proportions.")
        if s_jaw > 95: verdict.append("Strong, dominant jawline.")
        if s_tilt > 90: verdict.append("Positive canthal tilt (Hunter Eyes).")

        return {
            "overall": {
                "harmonyScore": round(overall_score, 1),
                "symmetryScore": round(s_symmetry, 1),
                "verdict": " ".join(verdict)
            },
            "proportions": {
                "phiRatio": round(phi_ratio, 3),
                "fifthsScore": round(s_fifths_match, 1),
                "thirds": {k: round(v, 1) for k, v in thirds.items()},
                "lowerThirdRatio": round(lower_ratio, 2),
                "fWHR": round(fwhr, 2) # Added fWHR explicitly
            },
            "eyes": {
                "canthalTilt": round(avg_eye_tilt, 1),
                "eyeSpacingRatio": round(esr, 3), 
                "eyeAreaRatio": round(eye_area_ratio, 3),
                "harmonyIndex": round(s_fifths_match, 1) 
            },
            "nose": {
                "mouthNoseRatio": round(mouth_nose_ratio, 2),
                "score": round(s_mouth_nose, 1),
                "noseWidthRatio": round(nose_w / bizygoma, 3),
                "symmetry": round(sym_nose, 1)
            },
            "jawline": {
                "gonialAngle": round((self._get_angle(p, "gonionLeft", "zygomaLeft", "menton") + self._get_angle(p, "gonionRight", "zygomaRight", "menton"))/2, 1),
                "jawToCheekRatio": round(bigonial/bizygoma, 2),
                "chinPhiltrumRatio": round(chin_philtrum_ratio, 2),
                "symmetry": round(sym_jaw, 1),
                "lowerThirdRatio": round(lower_ratio, 2)
            },
            "detailed": {
                "Rule of Fifths Match": f"{s_fifths_match:.1f}%",
                "Lower Third (1:2)": f"{lower_ratio:.2f}",
                "Midface Ratio (fWHR)": f"{fwhr:.2f}",
                "Golden Ratio Score": f"{s_phi:.1f}%",
                "Canthal Tilt Score": f"{s_tilt:.1f}%"
            }
        }

    def _poly_area(self, pts):
        x = [pt[0] for pt in pts]
        y = [pt[1] for pt in pts]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _get_angle(self, p, center, p1, p2):
        v1 = p[p1] - p[center]
        v2 = p[p2] - p[center]
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))

