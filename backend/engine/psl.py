import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any

class PSLEngine:
    """
    Refined PSL Inference Engine.
    Uses Mutual Implication (Equality Constraints) to ensure Harmony 
    is the weighted equilibrium of all features.
    """
    
    def __init__(self):
        # Adjusted Weights (Foundation-first)
        self.weights = {
            "symmetry": 12.0,      
            "proportions": 10.0,   
            "angles": 8.0,         
            "golden_ratio": 5.0,   
            "balance": 5.0
        }

    def infer_harmony(self, features: Dict[str, float]) -> Dict[str, Any]:
        obs = {k: max(0.0, min(1.0, v)) for k, v in features.items()}
        
        def objective(x):
            harmony = x[0]
            loss = 0.0
            
            # Equality Rules: w * (Harmony - Feature)^2
            for key, weight in self.weights.items():
                # Check mapping for keys from morphology
                feat_val = obs.get(key)
                if feat_val is not None:
                    loss += weight * (harmony - feat_val) ** 2

            return loss

        # Solve
        result = minimize(
            fun=objective,
            x0=[0.5],
            bounds=[(0.0, 1.0)],
            method='SLSQP'
        )
        
        final_harmony = float(result.x[0])
        
        # Explainability: Relative contribution to the final weighted average
        explanations = self._explain_score(obs, final_harmony)
        
        return {
            "harmony_score": final_harmony,
            "explanations": explanations,
            "optimization_success": result.success
        }

    def _explain_score(self, obs: Dict[str, float], final_score: float) -> Dict[str, float]:
        active_weights_sum = sum(w for k, w in self.weights.items())
        if active_weights_sum == 0: return {}
        
        explanations = {}
        for key in ["symmetry", "proportions", "angles", "golden_ratio", "balance"]:
            weight = self.weights.get(key, 0)
            val = obs.get(key, 0)
            # Contribution to the weighted average: (w * val) / Sum(w)
            explanations[key] = (weight * val) / active_weights_sum
            
        return explanations
