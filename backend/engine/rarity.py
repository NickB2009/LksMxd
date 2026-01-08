from scipy.stats import norm
import numpy as np

# Population Statistics (Global Average estimations)
POPULATION_STATS = {
    "fWHR": {"mean": 1.75, "std": 0.12},
    "facialIndex": {"mean": 1.35, "std": 0.1},
    "canthalTilt": {"mean": 2.0, "std": 3.0}, # Degrees
    "jawToCheek": {"mean": 0.78, "std": 0.08},
    "eyeSpacingRatio": {"mean": 1.0, "std": 0.15},
    "midFaceRatio": {"mean": 1.0, "std": 0.12}  # Mid/Lower
}

def calculate_rarity(metrics: dict) -> dict:
    total_rarity = 0
    count = 0
    details = {}
    
    for trait, stats in POPULATION_STATS.items():
        if trait not in metrics: continue
        
        val = metrics[trait]
        z = (val - stats["mean"]) / stats["std"]
        
        # Two-tailed probability
        p = 2 * (1 - norm.cdf(abs(z)))
        if p < 1e-6: p = 1e-6
        
        # Rarity Score: 1 (Common) to 10 (Unique)
        # Log scale: 10% -> score 2? 1% -> score 4?
        # Let's map z-score directly to 1-10 simpler?
        # Z=0 -> 1. Z=3 -> 10.
        rarity_score = min(10, 1 + abs(z) * 3)
        
        total_rarity += rarity_score
        count += 1
        
        details[trait] = {
            "value": val,
            "zScore": round(z, 2),
            "rarity": round(rarity_score, 1),
            "label": _label(rarity_score)
        }
        
    avg_score = total_rarity / count if count > 0 else 1
    
    return {
        "score": round(avg_score, 1),
        "label": _global_label(avg_score),
        "details": details
    }

def _label(s):
    if s > 7.5: return "Extremely Rare"
    if s > 5.0: return "Distinct"
    if s > 2.5: return "Uncommon"
    return "Typical"

def _global_label(s):
    if s > 6: return "High Distinctiveness"
    if s > 4: return "Standout Features"
    return "Balanced / Common Profile"
