import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import os
from typing import Dict, Any

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
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None: return None
        
        # Apply pre-processing for better landmark mapping
        processed_image = self._preprocess_image(image)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        detection_result = self.landmarker.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return None
            
        landmarks = detection_result.face_landmarks[0]
        h, w, _ = image.shape
        points = {k: np.array([landmarks[v].x * w, landmarks[v].y * h]) for k, v in LANDMARKS.items()}
        return self._calculate_metrics(points)

    def _calculate_metrics(self, p: Dict[str, np.ndarray]) -> Dict[str, Any]:
        def d(k1, k2): return np.linalg.norm(p[k1] - p[k2])
        
        # --- 1. BASIC DIMENSIONS ---
        bizygoma = d("zygomaLeft", "zygomaRight")
        face_h = d("trichion", "menton")
        bigonial = d("gonionLeft", "gonionRight")
        phi_ratio = face_h / bizygoma if bizygoma > 0 else 1.6
        fwhr = bizygoma / d("glabella", "lipTop") if d("glabella", "lipTop") > 0 else 1.9

        # --- 2. VERTICAL THIRDS (Moridani: Trichion-Glabella-Subnasale-Menton) ---
        u_h = d("trichion", "glabella")
        m_h = d("glabella", "subnasale")
        l_h = d("subnasale", "menton")
        tot_h = u_h + m_h + l_h
        thirds = {"upper": u_h/tot_h*100, "mid": m_h/tot_h*100, "lower": l_h/tot_h*100}
        
        # Lower Third Subdivision (Ideal 1:2 ratio: Subnasale-Stomion : Stomion-Menton)
        # using lipTop as reliable 'stomion' approx for closed mouth
        lower_upper = d("subnasale", "lipTop")
        lower_lower = d("lipBottom", "menton") 
        # Note: paper often uses 'stomion' (mouth center). We use lipTop/Bottom to define mass.
        # Adjusted: Subnasale to Lip Separation vs Lip Separation to Menton
        lower_ratio = lower_lower / lower_upper if lower_upper > 0 else 2.0
        
        # --- 3. HORIZONTAL RULE OF FIFTHS ---
        # Segments: 1. Left Ear-Eye, 2. Left Eye, 3. Inter-Eye, 4. Right Eye, 5. Right Eye-Ear
        # Proxies: We use zygoma as lateral face bounds because ears are unreliable in 2D
        w_face = bizygoma
        w_eye_l = d("eyeLeftInner", "eyeLeftOuter")
        w_eye_r = d("eyeRightInner", "eyeRightOuter")
        w_inters = d("eyeLeftInner", "eyeRightInner")
        # Outer segments (approximate using face width remainder)
        w_outer_l = (bizygoma - w_eye_l - w_inters - w_eye_r) / 2 # simplified assumption of symmetry for the breakdown
        
        # Rule of Fifths deviations (Ideal: all 5 segments roughly equal, or specific ratios)
        # Ideally Inter-eye width = Eye Width
        esr = w_inters / ((w_eye_l + w_eye_r)/2) # Eye Spacing Ratio

        # --- 4. EYES & GAZE ---
        tilt_l = (p["eyeLeftInner"][1] - p["eyeLeftOuter"][1]) / d("eyeLeftInner", "eyeLeftOuter") * 100
        tilt_r = (p["eyeRightInner"][1] - p["eyeRightOuter"][1]) / d("eyeRightInner", "eyeRightOuter") * 100
        avg_eye_tilt = (tilt_l + tilt_r) / 2
        
        eye_area_l = self._poly_area([p[k] for k in ["eyeLeftInner", "eyeLeftTop", "eyeLeftTopOuter", "eyeLeftOuter", "eyeLeftBottomOuter", "eyeLeftBottom"]])
        eye_area_r = self._poly_area([p[k] for k in ["eyeRightInner", "eyeRightTop", "eyeRightTopOuter", "eyeRightOuter", "eyeRightBottomOuter", "eyeRightBottom"]])
        eye_area_ratio = (eye_area_l + eye_area_r) / (face_h * bizygoma) * 100

        # --- 5. GOLDEN RATIOS (Φ) ---
        nose_w = d("noseAlareLeft", "noseAlareRight")
        mouth_w = d("mouthLeft", "mouthRight")
        mouth_nose_ratio = mouth_w / nose_w if nose_w > 0 else 1.618

        # --- 6. SCORING (Fuzzy Inference Simulation) ---
        def gaussian(x, mu, sigma):
            return 100 * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        # A. Proportions (Structure)
        s_thirds = 100 - (abs(thirds["upper"]-33.3) + abs(thirds["mid"]-33.3) + abs(thirds["lower"]-33.3))
        s_low_ratio = gaussian(lower_ratio, 2.0, 0.3) # Ideal 2.0
        s_fifths = gaussian(esr, 1.0, 0.15) # Inter-eye = Eye width => Ratio 1.0 (ESR ~0.46 in other metric, here comparing widths directly)
        
        # Standardize ESR for the report (width of inter/width of face)? 
        # Actually keeping previous ESR def: inter / biocular for consistency with user history, but scoring based on "Eye Width Match"
        # Let's align with Paper: Eye Separation Ratio (ESR) ideal is 0.46 of biocular, OR inter-eye distance = 1 eye width.
        # Use simple Eye Width matching for Fifths score.
        eye_width_avg = (w_eye_l + w_eye_r) / 2
        fifths_deviation = abs(w_inters - eye_width_avg) / eye_width_avg
        s_fifths_match = 100 * max(0, 1 - fifths_deviation)

        # B. Golden Ratio
        s_phi = gaussian(phi_ratio, 1.618, 0.1)
        s_mouth_nose = gaussian(mouth_nose_ratio, 1.618, 0.2)

        # C. Sexual Dimorphism (Jaw/Eye)
        # Men: wider jaw (ratio ~0.9), Hunter eyes (positive tilt)
        # Women: narrower jaw, neutral tilt. Assuming Neutral/Male bias for "Scout" context or Neutral?
        # Using broadly attractive standards (Dimorphism-neutral optimal ranges)
        s_jaw = gaussian(bigonial/bizygoma, 0.85, 0.1) 
        s_tilt = gaussian(avg_eye_tilt, 5.0, 4.0) # slightly positive is generally ideal

        # D. Symmetry
        midline_x = (p["glabella"][0] + p["menton"][0]) / 2
        def get_sym(kL, kR): 
            return 100 - (abs(abs(p[kL][0]-midline_x) - abs(p[kR][0]-midline_x)) / bizygoma * 500)
        
        sym_eyes = get_sym("eyeLeftOuter", "eyeRightOuter")
        sym_jaw = get_sym("gonionLeft", "gonionRight")
        sym_nose = get_sym("noseAlareLeft", "noseAlareRight")
        s_symmetry = (sym_eyes + sym_jaw + sym_nose) / 3

        # --- 7. AGGREGATION (Moridani Weights) ---
        # Structure > Features
        # "Fuzzy" aggregation:
        
        overall_score = (
            0.25 * s_fifths_match +   # Horizontal Balance
            0.20 * s_thirds +         # Vertical Balance
            0.15 * s_phi +            # Golden Ratio (Face)
            0.15 * s_symmetry +       # Global Symmetry
            0.15 * s_mouth_nose +     # Feature Relations
            0.10 * s_low_ratio        # Lower Face refinement
        )

        verdict = []
        if overall_score > 85: verdict.append("Excellent aesthetic harmony.")
        elif overall_score > 75: verdict.append("Strong facial structure.")
        if s_fifths_match > 90: verdict.append("Ideal horizontal proportions (Rule of Fifths).")
        if lower_ratio > 1.8 and lower_ratio < 2.2: verdict.append("Perfect lower facial third (1:2).")

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
                "eyeSpacingRatio": round(esr, 3), # Kept as Inter/EyeWidth for this version
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
                "symmetry": round(sym_jaw, 1)
            },
            "detailed": {
                "Rule of Fifths Match": f"{s_fifths_match:.1f}%",
                "Lower Third (1:2)": f"{lower_ratio:.2f}",
                "Golden Ratio Score": f"{s_phi:.1f}%",
                "Mouth-Nose Φ": f"{mouth_nose_ratio:.2f}"
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

