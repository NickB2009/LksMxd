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

        # --- 2. FACIAL THIRDS ---
        u_h = d("trichion", "glabella")
        m_h = d("glabella", "subnasale")
        l_h = d("subnasale", "menton")
        tot = u_h + m_h + l_h if (u_h + m_h + l_h) > 0 else 1
        thirds = {"upper": u_h/tot*100, "mid": m_h/tot*100, "lower": l_h/tot*100}

        # --- 3. EYES ANALYSIS (Refined) ---
        biocular_width = d("eyeLeftOuter", "eyeRightOuter")
        inner_canthal_dist = d("eyeLeftInner", "eyeRightInner")
        esr = inner_canthal_dist / biocular_width if biocular_width > 0 else 0.46
        
        # Stabilized Canthal Tilt (Using outer and inner points)
        tilt_l = (p["eyeLeftInner"][1] - p["eyeLeftOuter"][1]) / d("eyeLeftInner", "eyeLeftOuter") * 100
        tilt_r = (p["eyeRightInner"][1] - p["eyeRightOuter"][1]) / d("eyeRightInner", "eyeRightOuter") * 100
        avg_eye_tilt = (tilt_l + tilt_r) / 2
        
        # Precise Eye Area mapping (Approximate polygon)
        def eye_area(eye):
            # Points: Inner, TopInner, Top, TopOuter, Outer, BottomOuter, Bottom, BottomInner
            pts = [p[f"eye{eye}Inner"], p[f"eye{eye}TopInner"], p[f"eye{eye}Top"], 
                   p[f"eye{eye}TopOuter"], p[f"eye{eye}Outer"], p[f"eye{eye}BottomOuter"], 
                   p[f"eye{eye}Bottom"], p[f"eye{eye}BottomInner"]]
            # Shoelace formula for area
            x = [pt[0] for pt in pts]
            y = [pt[1] for pt in pts]
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        eye_l_area = eye_area("Left")
        eye_r_area = eye_area("Right")
        eye_area_ratio = (eye_l_area + eye_r_area) / (face_h * bizygoma)

        # Eye Symmetry
        eye_sym = 100 - abs(tilt_l - tilt_r) * 2

        # --- 4. NOSE ANALYSIS ---
        nose_l = d("nasion", "subnasale")
        nose_w = d("noseAlareLeft", "noseAlareRight")
        nose_ratio = nose_l / nose_w if nose_w > 0 else 1.5
        nose_to_face = nose_w / bizygoma if bizygoma > 0 else 0.25
        
        # Nose Symmetry
        midline_x = (p["nasion"][0] + p["menton"][0]) / 2
        nose_sym = 100 - abs(abs(p["noseAlareLeft"][0] - midline_x) - abs(p["noseAlareRight"][0] - midline_x)) / nose_w * 100

        # --- 5. JAWLINE & CHIN ---
        def get_angle(center, p1, p2):
            v1 = p[p1] - p[center]
            v2 = p[p2] - p[center]
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        angle_l = get_angle("gonionLeft", "zygomaLeft", "menton")
        angle_r = get_angle("gonionRight", "zygomaRight", "menton")
        gonial_avg = (angle_l + angle_r) / 2
        
        jaw_to_cheek = bigonial / bizygoma
        chin_h = d("menton", "lipLowerTop")
        philtrum = d("subnasale", "lipTop")
        chin_phil_ratio = chin_h / philtrum if philtrum > 0 else 2.0
        
        jaw_sym = 100 - abs(angle_l - angle_r) * 2

        # --- 6. HARMONY & OVERALL (Recalibrated based on research paper) ---
        # Symmetry index
        sym_score = (eye_sym * 0.4 + nose_sym * 0.3 + jaw_sym * 0.3) 
        
        # Feature-specific scores (following Part-Based approach)
        phi_score = 100 * np.exp(-0.5 * ((phi_ratio - 1.618) / 0.08)**2)
        esr_score = 100 * np.exp(-0.5 * ((esr - 0.47) / 0.025)**2)
        fwhr_score = 100 * np.exp(-0.5 * ((fwhr - 1.9) / 0.12)**2)
        thirds_score = 100 - (abs(thirds["upper"]-33.3) + abs(thirds["mid"]-33.3) + abs(thirds["lower"]-33.3))
        
        # Heavily weight the Eye area (primary predictor)
        eye_harmony = (esr_score * 0.7 + eye_sym * 0.3)
        
        # Final Harmony calculation with increased focus on Eye Harmony and Symmetry
        harmony_score = (eye_harmony * 0.35 + sym_score * 0.25 + phi_score * 0.15 + fwhr_score * 0.15 + thirds_score * 0.1)

        verdict = []
        if harmony_score > 88: verdict.append("High Aesthetic Harmony.")
        elif harmony_score > 80: verdict.append("Good facial balance.")
        
        if sym_score > 95: verdict.append("Exceptional Symmetry.")
        if gonial_avg < 125: verdict.append("Strongly defined jawline.")
        if nose_ratio > 1.6: verdict.append("Refined nasal proportions.")

        return {
            "overall": {
                "harmonyScore": round(harmony_score, 1),
                "symmetryScore": round(sym_score, 1),
                "verdict": " ".join(verdict)
            },
            "proportions": {
                "phiRatio": round(phi_ratio, 3),
                "fWHR": round(fwhr, 2),
                "thirds": {k: round(v, 1) for k, v in thirds.items()}
            },
            "eyes": {
                "canthalTilt": round(avg_eye_tilt, 1),
                "eyeSpacingRatio": round(esr, 3),
                "eyeAreaRatio": round(eye_area_ratio, 5),
                "harmonyIndex": round(eye_harmony, 1),
                "symmetry": round(eye_sym, 1)
            },
            "nose": {
                "noseRatio": round(nose_ratio, 2),
                "noseToFaceWidth": round(nose_to_face, 3),
                "symmetry": round(nose_sym, 1)
            },
            "jawline": {
                "gonialAngle": round(gonial_avg, 1),
                "jawToCheekRatio": round(jaw_to_cheek, 2),
                "chinPhiltrumRatio": round(chin_phil_ratio, 2),
                "symmetry": round(jaw_sym, 1)
            },
            "detailed": {
                "Symmetry Index": f"{sym_score:.1f}%",
                "Eye Harmony": f"{eye_harmony:.1f}%",
                "Gonial Angle": f"{gonial_avg:.1f}°",
                "Face Height": f"{int(face_h)}px"
            }
        }

