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
    # --- GLOBAL CONSTANTS ---
    CONSTANTS = {
        "EPS": 1e-6,
        "SYM_K": 18.0,
        "SYM_FLOOR": 0.25,
        "COHERENCE_LAMBDA": 1.2,
        
        # Population Est (Normalized)
        "POP_STATS": {
            "canthal_tilt": {"mu": 2.0, "sigma": 2.5},
            "gonial_angle": {"mu": 124.0, "sigma": 7.0},
            "fwhr": {"mu": 0.74, "sigma": 0.05}, # facial width / height
            "midface_ratio": {"mu": 0.33, "sigma": 0.04}
        },
        
        "MALE": {
            "angles": {
                "canthal_tilt": {"desirable": [0.0, 2.0], "bonus": [1.0, 3.0]}, # z-scores
                "gonial_angle": {"ideal": [115.0, 130.0], "bonus": [118.0, 125.0]} # degrees
            },
            "proportions": {
                "thirds_sigma": 0.075,
                "fifths_sigma": 0.085,
                "fwhr_bonus": [0.78, 0.85]
            },
            "distinctiveness": {
                "angles":    {"a1": 0.4, "a2": 1.0, "b2": 2.5, "b1": 3.2},
                "structure": {"a1": 0.3, "a2": 0.9, "b2": 2.3, "b1": 3.0},
                "ratios":    {"a1": 0.2, "a2": 0.8, "b2": 2.0, "b1": 2.8}
            }
        },
        
        "FEMALE": {
            "angles": {
                "canthal_tilt": {"desirable": [0.5, 2.5], "bonus": [1.2, 3.0]},
                "gonial_angle": {"ideal": [120.0, 135.0], "bonus": [125.0, 132.0]}
            },
            "proportions": {
                "thirds_sigma": 0.065,
                "fifths_sigma": 0.075,
                "fwhr_bonus": [0.70, 0.76]
            },
            "distinctiveness": {
                "angles":    {"a1": 0.3, "a2": 0.9, "b2": 2.0, "b1": 2.6},
                "structure": {"a1": 0.2, "a2": 0.7, "b2": 1.8, "b1": 2.4},
                "ratios":    {"a1": 0.2, "a2": 0.7, "b2": 1.6, "b1": 2.3}
            }
        }
    }

    def __init__(self, gender='male'):
        self.gender = gender.lower()
        self.params = self.CONSTANTS['MALE'] if self.gender == 'male' else self.CONSTANTS['FEMALE']
        
        # Initialize MediaPipe
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
        features["distinctiveness"] = self._compute_distinctiveness(points) # NEW: Structured Distinctiveness

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
        # exp(-k * error) with floor
        k = self.CONSTANTS['SYM_K']
        floor = self.CONSTANTS['SYM_FLOOR']
        
        raw_score = np.exp(-k * mean_error)
        return float(np.clip(raw_score, floor, 1.0))

    def _trapezoidal_score(self, z: float, win: Dict[str, float]) -> float:
        a1, a2, b2, b1 = win['a1'], win['a2'], win['b2'], win['b1']
        if z < a1 or z > b1: return 0.0
        if a2 <= z <= b2: return 1.0
        if a1 <= z < a2: return (z - a1) / (a2 - a1)
        if b2 < z <= b1: return (b1 - z) / (b1 - b2)
        return 0.0

    def _compute_distinctiveness(self, p: Dict[str, np.ndarray]) -> float:
        def d(k1, k2): return np.linalg.norm(p[k1][:2] - p[k2][:2])
        pop = self.CONSTANTS['POP_STATS']
        wins = self.params['distinctiveness']
        
        scores = []
        
        # 1. Structure (fWHR approx)
        bizygoma = d("zygomaLeft", "zygomaRight")
        upper_lip_h = d("glabella", "lipTop")
        raw_fwhr = bizygoma / upper_lip_h if upper_lip_h > 0 else pop['fwhr']['mu']
        z_fwhr = abs(raw_fwhr - pop['fwhr']['mu']) / pop['fwhr']['sigma']
        scores.append(self._trapezoidal_score(z_fwhr, wins['structure']))
        
        # 2. Angles (Tilt)
        def get_tilt(inner, outer):
            dy, dx = p[outer][1] - p[inner][1], p[outer][0] - p[inner][0]
            return np.degrees(np.arctan2(-dy, dx))
        raw_tilt = (get_tilt("eyeLeftInner", "eyeLeftOuter") + get_tilt("eyeRightInner", "eyeRightOuter")) / 2
        z_tilt = abs(raw_tilt - pop['canthal_tilt']['mu']) / pop['canthal_tilt']['sigma']
        scores.append(self._trapezoidal_score(z_tilt, wins['angles']))
        
        # 3. Ratios (Midface)
        # Simplified ratio check
        z_mid = 1.0 # Placeholder
        scores.append(self._trapezoidal_score(z_mid, wins['ratios']))
        
        raw_dist = np.mean(scores) if scores else 0.0
        
        # Anti-Deformity Coherence Penalty
        # Proxy: tilt asymmetry
        tilt_diff = abs(get_tilt("eyeLeftInner", "eyeLeftOuter") - get_tilt("eyeRightInner", "eyeRightOuter"))
        # Using lambda decay
        coherence_penalty = np.exp(-self.CONSTANTS['COHERENCE_LAMBDA'] * tilt_diff) 
        
        return float(np.clip(raw_dist * coherence_penalty, 0.0, 1.0))

    def _compute_proportions(self, p: Dict[str, np.ndarray]) -> float:
        def d(k1, k2): return np.linalg.norm(p[k1][:2] - p[k2][:2])
        def gauss(val, target, sigma): return np.exp(-(val - target)**2 / (2 * sigma**2))

        # Vertical Thirds
        u, m, l = d("trichion", "glabella"), d("glabella", "subnasale"), d("subnasale", "menton")
        total_h = u + m + l
        if total_h == 0: return 0.0
        
        sigma_t = self.params['proportions']['thirds_sigma']
        u_s = gauss(u/total_h, 1/3, sigma_t)
        m_s = gauss(m/total_h, 1/3, sigma_t)
        l_s = gauss(l/total_h, 1/3, sigma_t)
        thirds_score = np.mean([u_s, m_s, l_s])

        # Horizontal Fifths
        face_w = d("earLeft", "earRight")
        eye_w = (d("eyeLeftInner", "eyeLeftOuter") + d("eyeRightInner", "eyeRightOuter")) / 2
        sigma_f = self.params['proportions']['fifths_sigma']
        
        fifths_score = gauss(face_w, 5 * eye_w, sigma_f * (5 * eye_w)) if eye_w > 0 else 0
        
        return float(np.clip((0.6 * thirds_score + 0.4 * fifths_score), 0.0, 1.0))

    def _compute_golden_ratio(self, p: Dict[str, np.ndarray]) -> float:
        # Alignment Score (Phi)
        def d(k1, k2): return np.linalg.norm(p[k1][:2] - p[k2][:2])
        def gauss(val, target, sigma): return np.exp(-(val - target)**2 / (2 * sigma**2))
        
        PHI = 1.618
        sigma = 0.12 
        scores = []
        
        h, w = d("trichion", "menton"), d("zygomaLeft", "zygomaRight")
        if w > 0: scores.append(gauss(h/w, PHI, sigma))
            
        mw, nw = d("mouthLeft", "mouthRight"), d("noseAlareLeft", "noseAlareRight")
        if nw > 0: scores.append(gauss(mw/nw, PHI, sigma))
            
        return float(np.clip(np.mean(scores), 0.0, 1.0)) if scores else 0.0

    def _compute_angles(self, p: Dict[str, np.ndarray]) -> float:
        # FASHION TRAIT SCORING (Bonus Zones)
        def plateau_gauss_skew(val, low, high, sigma_left, sigma_right):
            if low <= val <= high: return 1.0
            if val < low: return np.exp(-(val - low)**2 / (2 * sigma_left**2))
            return np.exp(-(val - high)**2 / (2 * sigma_right**2))

        def get_angle_deg(c, p1, p2):
            v1, v2 = p[p1][:2] - p[c][:2], p[p2][:2] - p[c][:2]
            return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))

        # 1. Gonial Angle
        # Broad acceptance (110-135), but punish extreme squareness (<100) hard, relax upper
        jaw_l = get_angle_deg("gonionLeft", "zygomaLeft", "menton")
        jaw_r = get_angle_deg("gonionRight", "zygomaRight", "menton")
        # Bonus zone: 110-130. Falloff slow above 130, fast below 110.
        s_jaw = plateau_gauss_skew((jaw_l + jaw_r) / 2, 110, 130, 20.0, 30.0) 
        
        # 2. Canthal Tilt
        # Bonus zone: +2 to +8 degrees (Hunter eyes / Foxy eyes)
        # Neutral (0) is okay (score ~0.8), Negative is bad.
        def get_tilt(inner, outer):
            dy, dx = p[outer][1] - p[inner][1], p[outer][0] - p[inner][0]
            return np.degrees(np.arctan2(-dy, dx))

        tilt_l, tilt_r = get_tilt("eyeLeftInner", "eyeLeftOuter"), get_tilt("eyeRightInner", "eyeRightOuter")
        # Skew right: forgiving valid positive tilts up to 12 deg. Punish negative.
        s_tilt = plateau_gauss_skew((tilt_l + tilt_r) / 2, 2.0, 8.0, 5.0, 8.0)

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
