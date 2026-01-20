"""
Score Fusion Module for PARALLEL mode.
Combines Face Recognition and Anti-Spoofing scores.
"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class FusionMethod(str, Enum):
    """Score fusion methods."""
    AND_GATE = "AND_GATE"
    WEIGHTED = "WEIGHTED"


@dataclass
class FusionResult:
    """Result of score fusion."""
    accept: bool
    reason: str
    score_final: Optional[float]
    details: str


class ScoreFusion:
    """
    Score Fusion module for PARALLEL mode.
    
    Supports two fusion methods:
    1. AND-GATE: Accept if both scores exceed thresholds
    2. WEIGHTED: Combine scores with weights, apply joint threshold
    """
    
    def __init__(
        self,
        method: FusionMethod = FusionMethod.AND_GATE,
        t_fas: float = 0.5,
        t_fr: float = 0.4,
        t_final: float = 0.6,
        t_fas_min: float = 0.3,
        w1: float = 0.6,  # Weight for FR score
        w2: float = 0.4   # Weight for FAS score
    ):
        """
        Initialize score fusion.
        
        Args:
            method: Fusion method (AND_GATE or WEIGHTED)
            t_fas: Anti-spoofing threshold
            t_fr: Face recognition threshold
            t_final: Final score threshold (for WEIGHTED)
            t_fas_min: Minimum FAS score (safety net for WEIGHTED)
            w1: Weight for FR score
            w2: Weight for FAS score
        """
        self.method = method
        self.t_fas = t_fas
        self.t_fr = t_fr
        self.t_final = t_final
        self.t_fas_min = t_fas_min
        self.w1 = w1
        self.w2 = w2
    
    def _normalize_fr_score(self, score: float) -> float:
        """
        Normalize FR score to [0, 1] range.
        FR cosine similarity is typically in [0, 1] but can be negative.
        
        Args:
            score: Raw FR score
            
        Returns:
            Normalized score in [0, 1]
        """
        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))
    
    def _and_gate_fusion(
        self,
        score_fr: float,
        score_fas: float
    ) -> FusionResult:
        """
        AND-GATE fusion: Both scores must exceed their thresholds.
        
        Args:
            score_fr: Face recognition score
            score_fas: Anti-spoofing score
            
        Returns:
            FusionResult
        """
        fas_pass = score_fas >= self.t_fas
        fr_pass = score_fr >= self.t_fr
        
        if not fas_pass:
            return FusionResult(
                accept=False,
                reason="SPOOF",
                score_final=None,
                details=f"FAS score {score_fas:.3f} < threshold {self.t_fas}"
            )
        
        if not fr_pass:
            return FusionResult(
                accept=False,
                reason="NON_MATCH",
                score_final=None,
                details=f"FR score {score_fr:.3f} < threshold {self.t_fr}"
            )
        
        return FusionResult(
            accept=True,
            reason="MATCHED",
            score_final=None,
            details=f"Both checks passed: FAS={score_fas:.3f}, FR={score_fr:.3f}"
        )
    
    def _weighted_fusion(
        self,
        score_fr: float,
        score_fas: float
    ) -> FusionResult:
        """
        WEIGHTED fusion: Combine scores with weights.
        
        Formula: score_final = w1 * normalize(score_FR) + w2 * score_FAS
        Accept if: score_final >= T_final AND score_FAS >= T_fas_min
        
        Args:
            score_fr: Face recognition score
            score_fas: Anti-spoofing score
            
        Returns:
            FusionResult
        """
        # Safety check: minimum FAS requirement
        if score_fas < self.t_fas_min:
            return FusionResult(
                accept=False,
                reason="SPOOF",
                score_final=None,
                details=f"FAS score {score_fas:.3f} < minimum threshold {self.t_fas_min}"
            )
        
        # Calculate weighted score
        normalized_fr = self._normalize_fr_score(score_fr)
        score_final = self.w1 * normalized_fr + self.w2 * score_fas
        
        if score_final < self.t_final:
            # Determine primary reason
            if normalized_fr < self.t_fr:
                reason = "NON_MATCH"
            else:
                reason = "LOW_CONFIDENCE"
            
            return FusionResult(
                accept=False,
                reason=reason,
                score_final=score_final,
                details=f"Final score {score_final:.3f} < threshold {self.t_final}"
            )
        
        return FusionResult(
            accept=True,
            reason="MATCHED",
            score_final=score_final,
            details=f"Weighted fusion passed: final={score_final:.3f}"
        )
    
    def fuse(
        self,
        score_fr: float,
        score_fas: float
    ) -> FusionResult:
        """
        Fuse FR and FAS scores.
        
        Args:
            score_fr: Face recognition score
            score_fas: Anti-spoofing score
            
        Returns:
            FusionResult with accept decision
        """
        if self.method == FusionMethod.AND_GATE:
            return self._and_gate_fusion(score_fr, score_fas)
        else:
            return self._weighted_fusion(score_fr, score_fas)
    
    def update_config(
        self,
        method: Optional[FusionMethod] = None,
        t_fas: Optional[float] = None,
        t_fr: Optional[float] = None,
        t_final: Optional[float] = None,
        t_fas_min: Optional[float] = None,
        w1: Optional[float] = None,
        w2: Optional[float] = None
    ):
        """Update fusion configuration."""
        if method is not None:
            self.method = method
        if t_fas is not None:
            self.t_fas = t_fas
        if t_fr is not None:
            self.t_fr = t_fr
        if t_final is not None:
            self.t_final = t_final
        if t_fas_min is not None:
            self.t_fas_min = t_fas_min
        if w1 is not None:
            self.w1 = w1
        if w2 is not None:
            self.w2 = w2


# Global instance
_fusion_instance: Optional[ScoreFusion] = None


def get_score_fusion(
    method: FusionMethod = FusionMethod.AND_GATE,
    t_fas: float = 0.5,
    t_fr: float = 0.4,
    t_final: float = 0.6,
    t_fas_min: float = 0.3,
    w1: float = 0.6,
    w2: float = 0.4
) -> ScoreFusion:
    """Get or create score fusion singleton."""
    global _fusion_instance
    if _fusion_instance is None:
        _fusion_instance = ScoreFusion(
            method=method,
            t_fas=t_fas,
            t_fr=t_fr,
            t_final=t_final,
            t_fas_min=t_fas_min,
            w1=w1,
            w2=w2
        )
    return _fusion_instance
