import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any

class PSLEngine:
    """
    Structured Distinctiveness PSL Engine.
    
    Logic paths:
    1. Natural Harmony: Alignment & Coherence -> Harmony
    2. Expressive Harmony: Distinctiveness & Coherence -> Harmony
    3. Chaos Penalty: Distinctiveness & !Coherence -> !Harmony
    """
    
    def __init__(self):
        # Rule Weights
        self.rules = {
            "natural_harmony": 6.0,     # Basic attractiveness
            "expressive_harmony": 9.0,  # Model-tier bonus
            "chaos_penalty": 11.0       # Anti-deformity safeguard
        }

    def infer_harmony(self, features: Dict[str, float]) -> Dict[str, Any]:
        obs = {k: max(0.0, min(1.0, v)) for k, v in features.items()}
        
        # 1. Base Predicates
        symmetry = obs.get("symmetry", 0.5)
        coherence = symmetry # Coherence proxy
        distinctiveness = obs.get("distinctiveness", 0.0)
        
        # Alignment Group
        alignment_feats = [obs.get("proportions", 0), obs.get("golden_ratio", 0), obs.get("balance", 0)]
        alignment = np.mean(alignment_feats) if alignment_feats else 0.0
        
        # Structure/Angle Quality (HighAngles & HighStructure)
        structure_quality = obs.get("angles", 0.0)

        # 2. Logic Paths (Hierarchical Aggregation)
        
        # Natural Harmony Calculation
        # Rule A: Alignment & Coherence -> Natural (Wt 7)
        # Rule B: Symmetry -> Natural (Wt 5)
        # We model Natural as the weighted mean of these inputs
        val_alignment_coherence = alignment * coherence
        phi_natural = (7.0 * val_alignment_coherence + 5.0 * symmetry) / 12.0
        
        # Expressive Harmony Calculation
        # Rule A: Distinctiveness & Coherence -> Expressive (Wt 9) # Primary driver (changed from 10 to 9 to match code edit)
        # Rule B: HighAngles & Structure -> Expressive (Wt 6)      # Secondary support
        # Note: Distinctiveness is the raw deviation magnitude. Coherence ensures it's not deformity.
        val_dist_coherence = distinctiveness * coherence
        phi_expressive = (9.0 * val_dist_coherence + 6.0 * structure_quality) / 15.0
        
        # Chaos Rule
        # Distinctiveness without Coherence -> Chaos IMPLIES Low Harmony
        phi_chaos = distinctiveness * (1.0 - coherence)

        # 3. Global Synthesis Optimization
        # Rule 1: Natural -> Global (Wt 6)
        # Rule 2: Expressive -> Global (Wt 9)
        # Rule 3: Chaos -> !Global (Wt 11)
        
        def objective(x):
            H = x[0]
            loss = 0.0
            loss += 6.0 * (max(0, phi_natural - H) ** 2)
            loss += 9.0 * (max(0, phi_expressive - H) ** 2)
            # Chaos Penalty: Violation if H > (1 - Chaos)
            loss += 11.0 * (max(0, H - (1.0 - phi_chaos)) ** 2)
            return loss

        result = minimize(objective, x0=[0.5], bounds=[(0.0, 1.0)], method='SLSQP')
        final_harmony = float(result.x[0])
        
        # 4. Explainability & Output
        return {
            "harmony_score": final_harmony,
            "dual_scores": {
                "natural": phi_natural,
                "expressive": phi_expressive,
                "chaos": phi_chaos
            },
            "explanations": self._explain_forces(phi_natural, phi_expressive, phi_chaos),
            "components": {
                "natural": phi_natural,
                "expressive": phi_expressive,
                "chaos": phi_chaos,
                "coherence": coherence,
                "alignment": alignment,
                "distinctiveness": distinctiveness
            }
        }

    def _explain_forces(self, natural, expressive, chaos):
        """
        Decomposes the score into the three logical forces.
        """
        # Normalize weights for display relativity
        total = self.rules["natural_harmony"] + self.rules["expressive_harmony"] + self.rules["chaos_penalty"]
        
        return {
            "Natural Harmony": (self.rules["natural_harmony"] * natural) / total,
            "Expressive Bonus": (self.rules["expressive_harmony"] * expressive) / total,
            "Chaos Penalty": -(self.rules["chaos_penalty"] * chaos) / total 
        }
