import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import os
import io
from typing import Dict, Any, List
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
    "lipTop": 0, "lipBottom": 17,
    "lipUpperInner": 13, "lipLowerInner": 14,

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
        Applies CLAHE and Gamma Correction to normalize lighting.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        processed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        gamma = 1.2
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(processed, table)

    def process_image(self, image_bytes: bytes) -> Dict[str, Any]:
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            pil_image = ImageOps.exif_transpose(pil_image)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            image = np.array(pil_image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"PIL Load Failed: {e}, falling back to cv2 raw decode")
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not decode image")
            
        processed_image = self._preprocess_image(image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        detection_result = self.landmarker.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return None
            
        landmarks_raw = detection_result.face_landmarks[0] # List[NormalizedLandmark]
        h, w, _ = image.shape
        
        # Convert to numpy Dict
        points = {k: np.array([landmarks_raw[v].x * w, landmarks_raw[v].y * h, landmarks_raw[v].z * w]) for k, v in LANDMARKS.items()}
        
        # --- MEASUREMENT LAYER ---
        features = {}
        features["symmetry"] = self._compute_symmetry(landmarks_raw, w, h)
        features["proportions"] = self._compute_proportions(points)
        features["golden_ratio"] = self._compute_golden_ratio(points)
        features["angles"] = self._compute_angles(points)
        features["balance"] = self._compute_balance(points)

        # Output structure
        lm_list = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks_raw]
        
        return {
            "features": features,
            "landmarks": lm_list
        }

    # --- GEOMETRIC FEATURE COMPUTATION ---

    def _compute_symmetry(self, landmarks, w, h) -> float:
        pairs = [
            (33, 263), (133, 362), (61, 291), (58, 288), (234, 454), (70, 300), (129, 358)
        ]
        mid_points = [10, 168, 6, 1, 164, 152]
        mid_x = np.mean([landmarks[i].x for i in mid_points])
        
        errors = []
        for l_idx, r_idx in pairs:
            l, r = landmarks[l_idx], landmarks[r_idx]
            l_mirrored_x = mid_x + (mid_x - l.x)
            dist = np.sqrt((l_mirrored_x - r.x)**2 + (l.y - r.y)**2)
            errors.append(dist)
            
        mean_error = np.mean(errors)
        # k=100.0: 1% error (0.01) -> exp(-1) = 0.36. Very strict.
        k = 100.0 
        score = np.exp(-k * mean_error)
        return float(np.clip(score, 0.0, 1.0))

    def _compute_proportions(self, p: Dict[str, np.ndarray]) -> float:
        def d(k1, k2): return np.linalg.norm(p[k1][:2] - p[k2][:2])
        def gauss(val, target, sigma): return np.exp(-(val - target)**2 / (2 * sigma**2))

        # Vertical Thirds
        u, m, l = d("trichion", "glabella"), d("glabella", "subnasale"), d("subnasale", "menton")
        total_h = u + m + l
        if total_h == 0: return 0.0
        
        # Sigma 0.03 is strictly defined for thirds (3% deviation)
        u_s = gauss(u/total_h, 1/3, 0.03)
        m_s = gauss(m/total_h, 1/3, 0.03)
        l_s = gauss(l/total_h, 1/3, 0.03)
        thirds_score = np.mean([u_s, m_s, l_s])

        # Horizontal Fifths
        face_w = d("earLeft", "earRight")
        eye_w = (d("eyeLeftInner", "eyeLeftOuter") + d("eyeRightInner", "eyeRightOuter")) / 2
        
        # dev = (face_w - 5*eye_w) / (5*eye_w)
        # gauss at 0 error with sigma 0.08
        fifths_score = gauss(face_w, 5 * eye_w, 0.1 * (5 * eye_w)) if eye_w > 0 else 0
        
        return float(np.clip((0.6 * thirds_score + 0.4 * fifths_score), 0.0, 1.0))

    def _compute_golden_ratio(self, p: Dict[str, np.ndarray]) -> float:
        def d(k1, k2): return np.linalg.norm(p[k1][:2] - p[k2][:2])
        def gauss(val, target, sigma): return np.exp(-(val - target)**2 / (2 * sigma**2))
        
        PHI = 1.618
        sigma = 0.08
        scores = []
        
        h, w = d("trichion", "menton"), d("zygomaLeft", "zygomaRight")
        if w > 0: scores.append(gauss(h/w, PHI, sigma))
            
        mw, nw = d("mouthLeft", "mouthRight"), d("noseAlareLeft", "noseAlareRight")
        if nw > 0: scores.append(gauss(mw/nw, PHI, sigma))
            
        nos_lip, lip_chin = d("subnasale", "lipTop"), d("lipBottom", "menton")
        if nos_lip > 0: scores.append(gauss(lip_chin/nos_lip, PHI, sigma))
            
        return float(np.clip(np.mean(scores), 0.0, 1.0)) if scores else 0.0

    def _compute_angles(self, p: Dict[str, np.ndarray]) -> float:
        def gauss(val, target, sigma): return np.exp(-(val - target)**2 / (2 * sigma**2))
        def get_angle_deg(c, p1, p2):
            v1, v2 = p[p1][:2] - p[c][:2], p[p2][:2] - p[c][:2]
            return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))

        # 1. Gonial Angle (Target ~122 deg, strict sigma 8)
        jaw_l = get_angle_deg("gonionLeft", "zygomaLeft", "menton")
        jaw_r = get_angle_deg("gonionRight", "zygomaRight", "menton")
        s_jaw = gauss((jaw_l + jaw_r) / 2, 122, 8)
        
        # 2. Canthal Tilt (Target 6 deg for 'Models', sigma 3)
        def get_tilt(inner, outer):
            dy, dx = p[outer][1] - p[inner][1], p[outer][0] - p[inner][0]
            return np.degrees(np.arctan2(-dy, dx))

        tilt_l, tilt_r = get_tilt("eyeLeftInner", "eyeLeftOuter"), get_tilt("eyeRightInner", "eyeRightOuter")
        s_tilt = gauss((tilt_l + tilt_r) / 2, 6, 3) 

        return float(np.clip((0.5 * s_jaw + 0.5 * s_tilt), 0.0, 1.0))

    def _compute_balance(self, p: Dict[str, np.ndarray]) -> float:
        """
        Area based balance (Left vs Right mass).
        """
        def poly_area(keys):
            pts = [p[k][:2] for k in keys]
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            
        # Cheek/Jaw/Eye areas
        area_l = poly_area(["eyeLeftTop", "eyeLeftBottom", "zygomaLeft", "gonionLeft", "chinLeft"])
        area_r = poly_area(["eyeRightTop", "eyeRightBottom", "zygomaRight", "gonionRight", "chinRight"])
        
        if area_l + area_r == 0: return 0.0
        
        # Balance ratio: closer to 1 is better
        ratio = min(area_l, area_r) / max(area_l, area_r)
        
        return float(np.clip(ratio, 0.0, 1.0))

