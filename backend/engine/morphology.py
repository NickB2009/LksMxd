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
        
        # --- EXTRAPOLATE FOREHEAD (Fix for "Stopping in middle") ---
        # Vector from Glabella (168) to Trichion (10)
        # We add points ABOVE Trichion to ensure visual coverage of high hairlines
        try:
            g = landmarks[168]
            t = landmarks[10]
            dx = t.x - g.x
            dy = t.y - g.y
            
            # Add 3 rows of points above the hairline
            for i in range(1, 4):
                scale = 0.5 * i
                lm_list.append({
                    "x": t.x + dx * scale,
                    "y": t.y + dy * scale,
                    "z": t.z
                })
                # Add side forehead points (temples) approximation
                lm_list.append({"x": t.x + dx*scale - 0.15*i, "y": t.y + dy*scale, "z": t.z})
                lm_list.append({"x": t.x + dx*scale + 0.15*i, "y": t.y + dy*scale, "z": t.z})
        except:
            pass # Safety if landmarks missing

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

        # --- 6. SCORING (HERO CURVES) ---
        def hero_score(x, min_v, max_v, sigma, penalize_excess=True):
            """
            Scores 100% if x is between min_v and max_v.
            Gaussian decay outside this range.
            If penalize_excess is False, values > max_v are treated as ideal (100%).
            """
            if min_v <= x <= max_v: return 100.0
            if not penalize_excess and x > max_v: return 100.0
            
            target = min_v if x < min_v else max_v
            return 100 * np.exp(-0.5 * ((x - target) / sigma) ** 2)

        # A. Proportions (Balance is strict)
        s_thirds = hero_score(abs(thirds["upper"]-thirds["lower"]), 0, 5, 5) # Score delta between thirds 
        s_low_ratio = hero_score(lower_ratio, 1.8, 2.2, 0.4) 
        s_fifths_match = fifths_score 
        
        # B. Golden Ratio
        s_phi = hero_score(phi_ratio, 1.55, 1.65, 0.1)
        s_mouth_nose = hero_score(mouth_nose_ratio, 1.5, 1.7, 0.25)

        # C. Sexual Dimorphism (Hero Traits)
        # Jaw: Wide relative to cheeks is good (Model tier). 0.85 - 1.1+
        s_jaw = hero_score(bigonial/bizygoma, 0.82, 1.1, 0.15, penalize_excess=False)
        
        # Tilt: Positive (Hunter) is good. 3 deg to 15 deg.
        s_tilt = hero_score(avg_eye_tilt, 3.0, 15.0, 5.0, penalize_excess=False)

        # D. Symmetry (Strict)
        midline_x = (p["glabella"][0] + p["menton"][0]) / 2
        def get_sym(kL, kR): 
            return 100 - (abs(abs(p[kL][0]-midline_x) - abs(p[kR][0]-midline_x)) / bizygoma * 500)
        
        sym_eyes = get_sym("eyeLeftOuter", "eyeRightOuter")
        sym_jaw = get_sym("gonionLeft", "gonionRight")
        sym_nose = get_sym("noseAlareLeft", "noseAlareRight")
        s_symmetry = (sym_eyes + sym_jaw + sym_nose) / 3

        # --- 7. AGGREGATION ---
        
        overall_score = (
            0.20 * s_fifths_match +   
            0.20 * s_thirds +         
            0.15 * s_phi +            
            0.15 * s_symmetry +       
            0.15 * s_mouth_nose +     
            0.15 * s_jaw # Increased Jaw Weight for models       
        )

        verdict = []
        if overall_score > 90: verdict.append("Elite Model Tier Aesthetics.")
        elif overall_score > 80: verdict.append("High Aesthetic Harmony.")
        
        if s_fifths_match > 90: verdict.append("Ideal horizontal proportions.")
        if s_jaw > 95: verdict.append("Strong, dominant jawline.")

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
                "lowerThirdRatio": round(lower_ratio, 2)
            },
            "eyes": {
                "canthalTilt": round(avg_eye_tilt, 1),
                "eyeSpacingRatio": round(esr, 3), 
                "eyeAreaRatio": round(eye_area_ratio, 3),
                "harmonyIndex": round(s_fifths_match, 1) # Utilizing fifths score here
            },
            "nose": {
                "mouthNoseRatio": round(mouth_nose_ratio, 2),
                "score": round(s_mouth_nose, 1)
            },
            "jawline": {
                "gonialAngle": round((self._get_angle(p, "gonionLeft", "zygomaLeft", "menton") + self._get_angle(p, "gonionRight", "zygomaRight", "menton"))/2, 1),
                "jawToCheekRatio": round(bigonial/bizygoma, 2),
                "symmetry": round(sym_jaw, 1),
                "lowerThirdRatio": round(lower_ratio, 2) # Added for FeatureCard
            },
            "detailed": {
                "Rule of Fifths Match": f"{s_fifths_match:.1f}%",
                "Lower Third (1:2)": f"{lower_ratio:.2f}",
                "Golden Ratio Score": f"{s_phi:.1f}%",
                "Sideburns Width": f"{int(w_face_full)}px"
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

