
from typing import List, Dict, Tuple

class MetricsCalculator:
    @staticmethod
    def compute(y_true: List[str], y_pred_top5: List[List[str]], durations: List[float]) -> Dict:
        """
        State 9: Metric Calculation
        
        Args:
            y_true: List of true DDC codes
            y_pred_top5: List of prediction lists (each up to 5 predicted codes)
            durations: List of inference durations in seconds
        
        Returns:
            Dictionary containing Top-1, Top-3, Top-5 accuracy and timing metrics
        """
        total = len(y_true)
        if total == 0:
            return {
                'top1_accuracy': 0.0,
                'top3_accuracy': 0.0,
                'top5_accuracy': 0.0,
                'avg_duration_ms': 0.0,
                'total_samples': 0
            }
        
        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        
        for true_code, pred_list in zip(y_true, y_pred_top5):
            # Ensure pred_list is a list
            if not isinstance(pred_list, list):
                pred_list = [pred_list] if pred_list else []
            
            # Get top-k predictions
            top1_preds = pred_list[:1]
            top3_preds = pred_list[:3]
            top5_preds = pred_list[:5]
            
            # Check hits
            if true_code in top1_preds:
                top1_correct += 1
            if true_code in top3_preds:
                top3_correct += 1
            if true_code in top5_preds:
                top5_correct += 1
        
        metrics = {
            'top1_accuracy': top1_correct / total,
            'top3_accuracy': top3_correct / total,
            'top5_accuracy': top5_correct / total,
            'top1_correct': top1_correct,
            'top3_correct': top3_correct,
            'top5_correct': top5_correct,
            'total_samples': total
        }
        
        # Duration Metrics
        if durations:
            metrics['avg_duration_ms'] = (sum(durations) / len(durations)) * 1000
            metrics['total_duration_s'] = sum(durations)
        else:
            metrics['avg_duration_ms'] = 0.0
            metrics['total_duration_s'] = 0.0
        
        # Extended Metrics using sklearn
        try:
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            
            # Extract Top-1 predictions handling empty lists
            y_pred_top1 = [preds[0] if preds else "UNKNOWN" for preds in y_pred_top5]
            
            # Compute weighted metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred_top1, average='weighted', zero_division=0
            )
            acc = accuracy_score(y_true, y_pred_top1)
            
            metrics.update({
                'accuracy': acc,
                'weighted_precision': precision,
                'weighted_recall': recall,
                'weighted_f1': f1
            })
        except ImportError:
            print("Warning: sklearn not installed. Weighted metrics skipped.")
            metrics.update({
                'accuracy': metrics['top1_accuracy'],
                'weighted_precision': 0.0,
                'weighted_recall': 0.0,
                'weighted_f1': 0.0
            })

        return metrics
