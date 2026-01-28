import numpy as np
from typing import Dict, Any

class PSLEngine:
    """
    Deterministic Coherence Engine.
    
    Computes two latent variables:
    1. Harmony (H): Classical structure (Symmetry, Proportions, Balance)
    2. Expressiveness (E): Structured deviation (Angles, Distinctiveness)
    
    Final Score (A) is constrained by the Coherence between H and E.
    """
    
    def __init__(self):
        pass

    def infer_harmony(self, features: Dict[str, float]) -> Dict[str, Any]:
        obs = {k: max(0.0, min(1.0, v)) for k, v in features.items()}
        
        # 1. Latent Variable Computation
        # Harmony (H): Classical Structure
        # Weights: Sym 0.35, Prop 0.30, Bal 0.20, Gold 0.15
        h_score = (
            0.35 * obs.get("symmetry", 0.5) +
            0.30 * obs.get("proportions", 0.5) +
            0.20 * obs.get("balance", 0.5) +
            0.15 * obs.get("golden_ratio", 0.5)
        )
        
        # Expressiveness (E): Deviation & Angularity
        # Weights: Angles 0.55, Dist 0.45
        e_score = (
            0.55 * obs.get("angles", 0.5) +
            0.45 * obs.get("distinctiveness", 0.0)
        )
        
        # 2. Coherence Constraint
        # Asymmetric Logic:
        # If E > H (Expressiveness exceeds basic Structure), risk of Alien/Deformity. Strict penalty.
        # If H > E (Structure exceeds Expressiveness), risk of Dullness. Lenient penalty (or none).
        
        diff = e_score - h_score
        
        if diff > 0:
            # Chaos Risk: Expressive but not Harmonic
            # e.g. E=0.8, H=0.4 -> diff 0.4 -> exp(-3*0.4) = 0.3
            coherence = np.exp(-3.0 * diff)
        else:
            # Dullness Risk: Harmonic but not Expressive
            # e.g. H=0.8, E=0.0 -> diff -0.8.
            # We don't want to punish "Good Structure" too much. 
            # Use a very light penalty or none (e.g., 1.0).
            # Let's use slight decay to encourage *some* expressiveness for 100/100.
            coherence = np.exp(-0.2 * abs(diff))
        
        # 3. Attractiveness (A)
        # Base weighted combination gated by Coherence
        # Note: If E is 0 (Average), A becomes approx 0.55 * H. 
        # For H=0.8, A=0.44. Scored 44/100. 
        # To align with "Average = 55-65", we might need to scale the result.
        # However, following valid logic:
        # A = (0.55 H + 0.45 E) * C
        a_score = (0.55 * h_score + 0.45 * e_score) * coherence
        
        # 4. Hard Cap for Unbalanced Modes
        # "If E > H + 0.15, penalize to prevent ugly-but-spiky dominance"
        cap_penalty = 1.0
        if e_score > h_score + 0.15:
            a_score *= 0.85
            cap_penalty = 0.85
            
        final_score = float(np.clip(a_score, 0.0, 1.0))
        
        # 5. Output
        return {
            "harmony_score": final_score,
            "dual_scores": {
                "natural": h_score,
                "expressive": e_score,
                "coherence": float(coherence),
                "cap_penalty": cap_penalty
            },
            "explanations": {
                "Natural Harmony": h_score,
                "Expressive Bonus": e_score,
                "Coherence Factor": coherence
            }
        }

    def _explain_score(self, *args):
        # Deprecated logic removed
        pass
