import numpy as np

# Market Ideal Profiles
# Now focusing on "Golden Ratio" compliance for Commercial/Ideal
# And "Editorial" deviations.

MARKETS = {
  "ideal_golden": {
    "description": "Scientific Perfection. Adherence to Golden Ratio & Neoclassical Canons.",
    "bounty": {
      "phiRatio": {"mean": 1.618, "std": 0.12},    # Relaxed from 0.05
      "eyeSpacingRatio": {"mean": 1.0, "std": 0.15}, # Relaxed from 0.1
      "mouthNoseRatio": {"mean": 1.618, "std": 0.2},
      "fWHR": {"mean": 1.9, "std": 0.15} 
    }
  },
  "editorial": {
    "description": "High fashion. Values extreme dimorphism, long midfaces, and unique features.",
    "bounty": {
      "phiRatio": {"mean": 1.50, "std": 0.15},
      "jawToCheek": {"mean": 0.92, "std": 0.08},
      "canthalTilt": {"mean": 8.0, "std": 4.0},
      "fWHR": {"mean": 2.2, "std": 0.25},  # Tuned for Jordan Barrett (2.26)
      "midface_compactness": {"mean": 1.9, "std": 0.2}, 
      "nose_ratio": {"mean": 0.8, "std": 0.15} 
    }
  },
  "commercial": {
    "description": "Balanced, symmetrical, approachable.",
    "bounty": {
      "phiRatio": {"mean": 1.6, "std": 0.08},
      "eyeSpacingRatio": {"mean": 1.0, "std": 0.08},
      "jawToCheek": {"mean": 0.82, "std": 0.05},
      "midface_compactness": {"mean": 2.1, "std": 0.1} # Prefers compact
    }
  }
}

def gaussian(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2)

def calculate_market_fit(metrics: dict) -> dict:
    results = {}
    for market, profile in MARKETS.items():
        total_score = 0
        count = 0
        
        for trait, stats in profile["bounty"].items():
            if trait not in metrics: continue
            val = metrics[trait]
            score = gaussian(val, stats["mean"], stats["std"])
            total_score += score
            count += 1
            
        avg_score = (total_score / count) if count > 0 else 0
        results[market] = {
            "score": int(avg_score * 100),
            "description": profile["description"],
            "traits": profile["bounty"]
        }
    return results

def calculate_potential(metrics: dict) -> dict:
    if not metrics: return None
    
    # Simulate Leanness / Optimization
    # 1. Jaw Definition improves (+15%)
    # 2. Phi Ratio might shift slightly due to less cheek fat (bizygoma reduces -> Phi increases?)
    #    - If cheek fat loss reduces Bizygoma width, FaceHeight/Bizygoma INCREASES.
    #    - This usually moves people closer to 1.618 if they are wide (low ratio).
    
    projected = metrics.copy()
    
    # Jaw improves
    projected["jawToCheek"] = min(0.98, projected.get("jawToCheek", 0.75) * 1.15)
    
    # Cheek fat loss (Bizygoma reduction simulation)
    # If Bizygoma reduces by 2%, Phi Ratio increases by ~2%
    curr_phi = projected.get("phiRatio", 1.5)
    projected["phiRatio"] = curr_phi * 1.02
    
    # Recalculate
    curr_fit = calculate_market_fit(metrics)
    proj_fit = calculate_market_fit(projected)
    
    potential_analysis = {}
    for m in curr_fit:
        potential_analysis[m] = {
            "pre_score": curr_fit[m]["score"],
            "post_score": proj_fit[m]["score"],
            "gain": proj_fit[m]["score"] - curr_fit[m]["score"],
            "note": "Simulated low body fat & reduced puffiness."
        }
        
    return potential_analysis
