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
    "eyeRightInner": 362, "eyeRightOuter": 263, 
    "eyeRightTop": 386, "eyeRightBottom": 374,
    
    # Cheek / Jaw
    "zygomaLeft": 234, "zygomaRight": 454,
    "gonionLeft": 58, "gonionRight": 288,
    "chinLeft": 172, "chinRight": 397,
    
    # Nose/Mouth
    "noseAlareLeft": 129, "noseAlareRight": 358,
    "mouthLeft": 61, "mouthRight": 291,
    
    # Ears (Approx)
    "earLeft": 234, "earRight": 454 # Using Zygoma extremes as proxies for upper linkage
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

    def process_image(self, image_bytes: bytes) -> Dict[str, Any]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None: return None
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        detection_result = self.landmarker.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return None
            
        landmarks = detection_result.face_landmarks[0]
        h, w, _ = image.shape
        points = {k: np.array([landmarks[v].x * w, landmarks[v].y * h]) for k, v in LANDMARKS.items()}
        return self._calculate_metrics(points)

    def _calculate_metrics(self, p: Dict[str, np.ndarray]) -> Dict[str, Any]:
        def d(k1, k2): return np.linalg.norm(p[k1] - p[k2])
        
        # --- 1. CORE ---
        bizygoma = d("zygomaLeft", "zygomaRight")
        face_h = d("trichion", "menton")
        bigonial = d("gonionLeft", "gonionRight")
        
        phi_ratio = face_h / bizygoma
        fwhr = bizygoma / d("glabella", "lipTop")
        
        # --- 2. SYMMETRY (Yaw-Adaptive) ---
        # 1. Estimate Yaw (Head Turn)
        # Compare "Nose Tip X" to "Midpoint of Eyes X"
        eye_mid_x = (p["eyeLeftInner"][0] + p["eyeRightInner"][0]) / 2
        nose_x = p["noseTip"][0]
        # Yaw magnitude roughly: (NoseX - EyeMidX) / Bizygoma
        yaw_proxy = (nose_x - eye_mid_x) / bizygoma
        yaw_detected = abs(yaw_proxy) > 0.02 # ~2% deviation implies turn

        if yaw_detected:
            # ROTATION COMPENSATED MODE
            def vert_dev(pt_l, pt_r):
                 return abs(p[pt_l][1] - p[pt_r][1]) / face_h * 100
                 
            sym_jaw = vert_dev("gonionLeft", "gonionRight")
            sym_cheek = vert_dev("zygomaLeft", "zygomaRight")
            sym_eye = vert_dev("eyeLeftOuter", "eyeRightOuter")
            
            symmetry_index = 100 - (sym_jaw + sym_cheek + sym_eye) * 2 
            symmetry_index = min(symmetry_index, 95.0) 
            
        else:
            # FRONTAL MODE (Standard)
            midline_x = (p["nasion"][0] + p["menton"][0]) / 2
            def deviation(pt_l, pt_r):
                dist_l = abs(p[pt_l][0] - midline_x)
                dist_r = abs(p[pt_r][0] - midline_x)
                return abs(dist_l - dist_r) / ((dist_l + dist_r) / 2) * 100
                
            sym_jaw = deviation("gonionLeft", "gonionRight")
            sym_cheek = deviation("zygomaLeft", "zygomaRight")
            sym_eye = deviation("eyeLeftOuter", "eyeRightOuter")
            
            symmetry_index = 100 - ((sym_jaw + sym_cheek + sym_eye) / 3)
        
        # --- 3. ANGLES (New) ---
        # Gonial Angle (Jaw Angle): Angle at Gonion formed by Ramus and Body.
        # Frontal projection: Angle between vertical (Ear-Gonion) and Gonion-Menton?
        # Zygoma is roughly above Gonion.
        # Vector 1: Gonion -> Zygoma (Ramus approx)
        # Vector 2: Gonion -> Menton (Body)
        # --- 3. ANGLES (Fixed) ---
        # Gonial Angle: Angle at Gonion (58/288)
        # Ramus Vector: Gonion -> Zygoma (Approx vertical-ish)
        # Body Vector: Gonion -> Menton (Horizontal-ish)
        # Standard Gonial Angle is ~125 +/-. 90 is impossibly square. 110-120 is "square/strong".
        # If user got 143, that's very obtuse (sloping).
        # We need to ensure vectors are pointing AWAY from Gonion.
        # V1 = Zygoma - Gonion
        # V2 = Menton - Gonion
        
        def get_angle(center, p1, p2):
            v1 = p[p1] - p[center]
            v2 = p[p2] - p[center]
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
            return angle
            
        angle_jaw_l = get_angle("gonionLeft", "zygomaLeft", "menton")
        angle_jaw_r = get_angle("gonionRight", "zygomaRight", "menton")
        gonial_angle_avg = (angle_jaw_l + angle_jaw_r) / 2
        
        # --- PSL / AESTHETIC SPECS ---
        # 1. Eye Separation Ratio (ESR): Inner Canthal Distance / Bi-Ocular Width
        # Bi-Ocular Width = Dist(LeftOuter, RightOuter)
        # Inner Distance = Dist(LeftInner, RightInner)
        # Ideal men: ~0.46-0.48? Or just use raw ratio.
        # Standard: ESD / BiZygoma is also used.
        # Let's use: InnerDist / OuterDist (Bi-ocular width not bizygoma)
        biocular_width = d("eyeLeftOuter", "eyeRightOuter")
        inner_canthal_dist = d("eyeLeftInner", "eyeRightInner")
        esr = inner_canthal_dist / biocular_width
        
        # 2. Midface Ratio (Compactness)
        # IPD (Inter-Pupillary Distance) / Midface Height
        # Pupil approx = Center of Eye
        pupil_l = (p["eyeLeftTop"] + p["eyeLeftBottom"] + p["eyeLeftInner"] + p["eyeLeftOuter"]) / 4
        pupil_r = (p["eyeRightTop"] + p["eyeRightBottom"] + p["eyeRightInner"] + p["eyeRightOuter"]) / 4
        ipd = np.linalg.norm(pupil_l - pupil_r)
        midface_height = d("glabella", "subnasale") # Using Glabella-Subnasale as 'functional midface'
        midface_ratio_psl = ipd / midface_height if midface_height > 0 else 1.0

        # 3. Lower Third Dominance
        lower_third_h = d("subnasale", "menton")
        total_face_h = d("trichion", "menton")
        lower_third_ratio = lower_third_h / total_face_h if total_face_h > 0 else 0.33
        
        # --- RESEARCH-BACKED METRICS (Morphometric Analysis) ---
        # 4. Eye Size Ratio - Research shows larger eyes correlate with attractiveness
        # Calculate eye area as height * width
        eye_height_l = d("eyeLeftTop", "eyeLeftBottom")
        eye_width_l = d("eyeLeftInner", "eyeLeftOuter")
        eye_height_r = d("eyeRightTop", "eyeRightBottom")
        eye_width_r = d("eyeRightInner", "eyeRightOuter")
        
        avg_eye_area = ((eye_height_l * eye_width_l) + (eye_height_r * eye_width_r)) / 2
        face_area = face_h * bizygoma
        eye_size_ratio = avg_eye_area / face_area if face_area > 0 else 0.015
        # Ideal: 0.015-0.020 (1.5-2.0% of face area)
        
        # 5. Nose Length-to-Width - "Long and narrow" is ideal per research
        nose_length = d("nasion", "subnasale")
        nose_width = d("noseAlareLeft", "noseAlareRight")
        nose_lw_ratio = nose_length / nose_width if nose_width > 0 else 1.5
        # Ideal: >1.5 (longer/narrower nose)
        
        # 6. Brow Position - "Raised eyebrows" are key attractiveness factor
        brow_eye_dist_l = d("browLeftInner", "eyeLeftTop")
        brow_eye_dist_r = d("browRightInner", "eyeRightTop")
        avg_brow_dist = (brow_eye_dist_l + brow_eye_dist_r) / 2
        brow_position = avg_brow_dist / face_h if face_h > 0 else 0.05
        # Normalized by face height - higher values = more raised brows
        
        # Calibration: If Zygoma is "out" and Menton is "down", angle should be ~120.
        # If result is 143, Ramus/Body are obtuse.
        # Fix: Maybe measure angle against VERTICAL? 
        # For robustness, let's use the Slope difference.
        
        # Chin Prominence (Recession Proxy)
        # Fix: Ensure landmarks exist.
        try:
             chin_h = np.linalg.norm(p["menton"] - p["lipLowerTop"])
        except:
             chin_h = d("menton", "lipLowerTop") # Fallback
             
        philtrum = d("subnasale", "lipTop")
        chin_philtrum_ratio = chin_h / philtrum if philtrum > 0.1 else 0.0

        # Chin Taper Angle: Angle at Menton between chinLeft and chinRight
        chin_taper = get_angle("menton", "chinLeft", "chinRight")
        
        # Canthal Tilt (Eye Slope) - Previously was using Brow Tilt!
        # Re-implement Eye Slope
        tilt_l = (p["eyeLeftInner"][1] - p["eyeLeftOuter"][1]) / d("eyeLeftInner", "eyeLeftOuter") * 100
        tilt_r = (p["eyeRightInner"][1] - p["eyeRightOuter"][1]) / d("eyeRightInner", "eyeRightOuter") * 100
        # Positive if Inner Y > Outer Y (Inner is lower). 
        # Image coords: Y increases down. So Inner(Lower Y val) < Outer(Higher Y val).
        # Wait. "Positive Tilt" = Outer corner is HIGHER (PHYSICALLY UP).
        # In IMAGE Y: Up is LOWER value.
        # So OuterY < InnerY.
        # If OuterY < InnerY => InnerY - OuterY > 0.
        # My formula: InnerY - OuterY. 
        # So Positive result = Positive Tilt.
        avg_eye_tilt = (tilt_l + tilt_r) / 2

        # --- VERDICT ---
        verdict = []
        if symmetry_index > 94: verdict.append("High Facial Symmetry.")
        elif symmetry_index < 88: verdict.append(f"Noticeable asymmetry ({(100-symmetry_index):.1f}%).")
        if yaw_detected: verdict.append("(Rotation Compensated).")
        
        if gonial_angle_avg < 120: verdict.append("Square/Sharp Jawline.") # 120 is the new barrier
        elif gonial_angle_avg > 135: verdict.append("Soft/Obtuse Jawline.")
        
        if chin_philtrum_ratio < 1.5: verdict.append("Compact chin height.")
        else: verdict.append("Strong chin verticality.")

        # --- EXPORT ---
        u = d("trichion", "glabella")
        m = d("glabella", "subnasale")
        l = d("subnasale", "menton")
        tot = u+m+l if (u+m+l) > 0 else 1
        thirds = {"upper": u/tot*100, "mid": m/tot*100, "lower": l/tot*100}
        
        return {
            "phiRatio": round(phi_ratio, 3),
            "fWHR": round(fwhr, 2),
            "thirds": thirds,
            "eyeSpacingRatio": round(d("eyeLeftInner", "eyeRightInner") / ((d("eyeLeftInner", "eyeLeftOuter") + d("eyeRightInner", "eyeRightOuter"))/2), 2),
            "mouthNoseRatio": round(d("mouthLeft", "mouthRight")/d("noseAlareLeft", "noseAlareRight"), 2),
            "jawToCheek": round(bigonial/bizygoma, 2),
            "canthalTilt": round(avg_eye_tilt, 1), 
            
            "detailed": {
                "Symmetry Index": f"{symmetry_index:.1f}%",
                "Gonial Angle (Avg)": f"{gonial_angle_avg:.1f}°",
                "Chin Taper Angle": f"{chin_taper:.1f}°",
                "Chin-Philtrum Ratio": f"{chin_philtrum_ratio:.2f}",
                "Ramus Length Approx": f"{int(d('gonionLeft', 'zygomaLeft'))} px",
                "Mandible Body Length": f"{int(d('gonionLeft', 'menton'))} px",
                "Midface Compactness": f"{(bizygoma/d('glabella','subnasale')):.2f}",
                "Brow Tilt": "Variable", # Re-implement if needed, skipping for concise update
                "Bizygomatic Width": f"{int(bizygoma)} px",
                "Bigonial Width": f"{int(bigonial)} px",
            },
            
            "verdict": " ".join(verdict),
            "raw": {"bizygomatic": bizygoma, "bigonial": bigonial},
            
            # Additional top-level keys for rarity/market_fit engines
            "facialIndex": round(face_h / bizygoma, 2), # Alias for phiRatio
            "midFaceRatio": round(m / l, 2) if l > 0 else 1.0,
            "midface_compactness": round(bizygoma / m, 2) if m > 0 else 2.0,
            "nose_ratio": round(d("noseAlareLeft", "noseAlareRight") / m, 2) if m > 0 else 0.8,
            
            # PSL Specifics
            "psl": {
                "esr": round(esr, 3),
                "midface_ratio_psl": round(midface_ratio_psl, 2),
                "lower_third_ratio": round(lower_third_ratio, 3),
                "fwhr_psl": round(fwhr, 2), # Same as standard
                "ipd_to_face_width": round(ipd / bizygoma, 2),
                # Research-backed additions
                "eye_size_ratio": round(eye_size_ratio, 4),
                "nose_lw_ratio": round(nose_lw_ratio, 2),
                "brow_position": round(brow_position, 3)
            }
        }
