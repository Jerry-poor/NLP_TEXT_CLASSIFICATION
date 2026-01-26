
import re
import json
from typing import List, Dict, Tuple, Optional

class ResponseValidator:
    """
    Response Parsing & Validation (State 8)
    Supports JSON parsing with fallback to lenient matching.
    """
    
    def __init__(self):
        pass

    def parse_and_validate(self, raw_response: str, candidates: List[Dict]) -> List[Tuple[str, float]]:
        """
        Parse model response and map to candidate codes.
        Returns list of (predicted_code, confidence) tuples, up to 5.
        
        Logic:
        1. Attempt to extract and parse JSON block.
        2. If validation/parsing fails, fallback to line-based parsing (old method) and raw search.
        """
        predictions = []
        
        # Build lookup: lowercase label -> code
        label_to_code = {}
        for cand in candidates:
            label_lower = cand['label'].lower().strip()
            label_to_code[label_lower] = cand['code']
        
        # 1. Try JSON Parsing
        try:
            # Find JSON block (from first { to last })
            json_match = re.search(r'(\{.*\})', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)
                
                # Check for 'predictions' key
                preds_list = []
                if isinstance(data, dict):
                    if 'predictions' in data and isinstance(data['predictions'], list):
                        preds_list = data['predictions']
                    # Handle if the model returned just a list
                    elif 'predictions' not in data and isinstance(data, dict):
                         # Maybe keys are categories?
                         pass
                
                # Iterate through parsed list
                for item in preds_list:
                    if not isinstance(item, dict):
                         continue
                         
                    # Extract label and confidence
                    label = item.get('category', '') or item.get('label', '')
                    conf = item.get('confidence', 0.0) or item.get('score', 0.0)
                    
                    # Normalize confidence to 0-1
                    if conf > 1.0:
                        conf = conf / 100.0
                        
                    # Match label to code
                    if label:
                         clean_label = str(label).strip().lower()
                         matched_code = self._lenient_match(clean_label, label_to_code)
                         if matched_code and matched_code not in [p[0] for p in predictions]:
                             predictions.append((matched_code, float(conf)))
                             
                    if len(predictions) >= 5:
                        break
        except Exception:
            # JSON parsing failed, silently fall back
            pass
            
        if predictions:
            return predictions

        # 2. Fallback: Text Regex Parsing (Old Method)
        lines = raw_response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip if lines look like JSON syntax
            if any(x in line for x in ['{', '}', '"category"', '"predictions"']):
                 continue

            # Try to extract confidence if present
            confidence = 0.5  # Default
            conf_match = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
            if conf_match:
                try:
                    confidence = float(conf_match.group(1)) / 100.0
                except:
                    pass
            
            # Clean the line for label matching
            cleaned = re.sub(r'[:\-]\s*\d+(?:\.\d+)?\s*%?', '', line)
            cleaned = cleaned.strip().lower()
            cleaned = re.sub(r'^[\d\.\-\*\)\s]+', '', cleaned).strip()
            
            matched_code = self._lenient_match(cleaned, label_to_code)
            
            if matched_code and matched_code not in [p[0] for p in predictions]:
                predictions.append((matched_code, confidence))
                
            if len(predictions) >= 5:
                break
        
        # 3. Fallback: Raw Search
        if not predictions:
            predictions = self._fallback_raw_search(raw_response, candidates)
        
        return predictions

    def _lenient_match(self, cleaned_text: str, label_to_code: Dict[str, str]) -> Optional[str]:
        """
        Matching logic:
        1. Exact match
        2. cleaned.startswith(label)  
        3. label in cleaned (containment)
        """
        if not cleaned_text:
             return None
             
        for label_lower, code in label_to_code.items():
            if cleaned_text == label_lower:
                return code
            if cleaned_text.startswith(label_lower):
                return code
            if label_lower in cleaned_text:
                return code
        return None

    def _fallback_raw_search(self, raw_response: str, candidates: List[Dict]) -> List[Tuple[str, float]]:
        """
        Fallback: scan entire raw response for any candidate label.
        """
        predictions = []
        response_lower = raw_response.lower()
        
        # Sort candidates by label length (longer first) to avoid partial matches
        sorted_candidates = sorted(candidates, key=lambda x: len(x['label']), reverse=True)
        
        for cand in sorted_candidates:
            label_lower = cand['label'].lower().strip()
            if label_lower in response_lower:
                # Use a simple heuristic: assume 0.5 confidence if found in raw text
                if cand['code'] not in [p[0] for p in predictions]:
                    predictions.append((cand['code'], 0.5))
                if len(predictions) >= 5:
                    break
        
        return predictions

    def get_top1_prediction(self, predictions: List[Tuple[str, float]]) -> Tuple[str, float]:
        """Helper to get top-1 prediction."""
        if predictions:
            return predictions[0]
        return ("UNKNOWN", 0.0)
