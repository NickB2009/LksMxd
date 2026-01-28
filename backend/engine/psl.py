import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Any, Tuple

class PSLEngine:
    """
    Probabilistic Soft Logic (PSL) Inference Engine for Facial Harmony.
    
    This engine treats facial analysis as a convex optimization problem.
    It takes normalized geometric features (0.0 - 1.0) as observed predicates
    and infers the hidden 'FacialHarmony' predicate value that minimizes
    the weighted distance to satisfaction of defined logical rules.
    """
    
    def __init__(self):
        # Default Weights (can be tuned or learned)
        # Higher weight = stricter rule (more penalty if violated)
        self.weights = {
            "symmetry_harmony": 10.0,
            "proportion_harmony": 10.0,
            "golden_harmony": 6.0,
            "angle_harmony": 8.0,
            "balance_harmony": 5.0,
            "prior_low": 2.0  # Regularizer: Harmony tends to be low unless driven up by features
        }

    def infer_harmony(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Runs the PSL inference optimization.
        
        Args:
            features: Dictionary of normalized feature scores (0-1).
                      check 'backend/engine/morphology.py' for keys.
                      
        Returns:
            Dictionary containing:
            - harmony_score: float (0-1)
            - explanations: Dict[str, float] (contribution per rule)
        """
        
        # 1. Observed Predicates (Inputs)
        # Ensure values are clipped 0-1
        obs = {k: max(0.0, min(1.0, v)) for k, v in features.items()}
        
        # 2. Objective Function (Energy / Loss)
        # We want to minimize the weighted sum of squared rule violations (hinge loss)
        # Rule form: w * max(0, Body - Head)^2 
        # (Implication A -> B is satisfied when A <= B, so violation is max(0, A - B))
        
        def objective(x):
            # x[0] is our target variable: FacialHarmony
            harmony = x[0]
            loss = 0.0
            
            # Rule 1: Symmetry(P) -> Harmony(P)
            # Implies strictly: High symmetry demands high harmony
            # But low symmetry doesn't restrict harmony (technically). 
            # In PSL specifically for scoring, we often use two-sided similarity or just implication.
            # Using simple implication A->B: violation = max(0, A - B)
            
            # R1: Symmetry -> Harmony
            viol_sym = max(0, obs.get("symmetry", 0) - harmony)
            loss += self.weights["symmetry_harmony"] * (viol_sym ** 2)
            
            # R2: Proportion -> Harmony
            viol_prop = max(0, obs.get("proportions", 0) - harmony)
            loss += self.weights["proportion_harmony"] * (viol_prop ** 2)
            
            # R3: AngleHarmony -> Harmony
            viol_angle = max(0, obs.get("angles", 0) - harmony)
            loss += self.weights["angle_harmony"] * (viol_angle ** 2)
            
            # R4: GoldenRatio -> Harmony
            viol_gold = max(0, obs.get("golden_ratio", 0) - harmony)
            loss += self.weights["golden_harmony"] * (viol_gold ** 2)
            
            # R5: Balance -> Harmony
            viol_bal = max(0, obs.get("balance", 0) - harmony)
            loss += self.weights["balance_harmony"] * (viol_bal ** 2)
            
            # R6: Prior (Regularization)
            # Harmony -> 0 (Softly pull down if no evidence supports it being high)
            # Violation = max(0, Harmony - 0) = Harmony
            viol_prior = harmony 
            loss += self.weights["prior_low"] * (viol_prior ** 2)

            return loss

        # 3. Solve (Constrained Optimization)
        # Harmony must be in [0, 1]
        result = minimize(
            fun=objective,
            x0=[0.5], # Start valid
            bounds=[(0.0, 1.0)],
            method='SLSQP' 
        )
        
        final_harmony = float(result.x[0])
        
        # 4. Explainability (Decomposition)
        # Calculate how much each feature "pushed" the score up (or satisfied the rule).
        # Since we use a convex loss, we can approximate contribution by the weight * activation.
        # But a more user-friendly explanation is: "How much did this feature contribute to the final score?"
        # We'll normalize the positive inputs based on weights.
        
        explanations = self._explain_score(obs, final_harmony)
        
        return {
            "harmony_score": final_harmony,
            "explanations": explanations,
            "optimization_success": result.success
        }

    def _explain_score(self, obs: Dict[str, float], final_score: float) -> Dict[str, float]:
        """
        Decomposes the final score into feature contributions.
        Since it's a non-linear optimization, this is an approximation for visualization.
        """
        # We calculate the "potential" of each feature
        total_weight = sum(self.weights.values())
        contributors = {}
        
        # Normalize weights excluding prior for relative importance display
        active_weights_sum = (
            self.weights["symmetry_harmony"] + 
            self.weights["proportion_harmony"] + 
            self.weights["angle_harmony"] + 
            self.weights["golden_harmony"] +
            self.weights["balance_harmony"]
        )
        
        if active_weights_sum == 0: return {}

        # Rough contribution: Weighted Input / Total Weighted Sum
        # This is a linear approximation of the equilibrium state
        for key in ["symmetry", "proportions", "angles", "golden_ratio", "balance"]:
            w_key = f"{key}_harmony"
            if w_key not in self.weights: 
                # mapping fix for keys that don't match exactly
                if key == "proportions": w_key = "proportion_harmony"
                elif key == "golden_ratio": w_key = "golden_harmony"
                elif key == "angles": w_key = "angle_harmony"
            
            weight = self.weights.get(w_key, 0)
            val = obs.get(key, 0)
            
            # Contribution to the "Push"
            # Why: The optimization solves for H such that weighted pull from all features balances the pull from prior.
            # Effectively H ~= (Sum w_i * x_i) / (Sum w_i + w_prior)
            # So contribution of x_i is (w_i * x_i).
            
            contributors[key] = (weight * val) / (active_weights_sum + self.weights["prior_low"])
            
        return contributors
