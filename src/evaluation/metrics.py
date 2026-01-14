from sklearn.metrics import f1_score


def macro_f1(y_true, y_pred, n_classes: int) -> float:
    """
    Calculate macro-averaged F1 score.
    
    Args:
        y_true: List or array of true labels
        y_pred: List or array of predicted labels
        n_classes: Number of classes
        
    Returns:
        Macro-averaged F1 score
    """
    return float(f1_score(y_true, y_pred, average='macro', zero_division=0.0))
