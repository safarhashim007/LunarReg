"""Cycle error in reference pixel units; preserves V9 edge-coordinate convention."""
import numpy as np


def cycle_metrics(source,returned,forward_confidence,reverse_confidence,scale):
    if not np.isfinite(scale) or scale<=0: raise ValueError('Scale must be positive')
    error=np.linalg.norm(np.asarray(returned)-np.asarray(source),axis=1)*scale
    confidence=np.sqrt(np.clip(forward_confidence,0,1)*np.clip(reverse_confidence,0,1))
    return error,confidence,confidence/(1+error)
