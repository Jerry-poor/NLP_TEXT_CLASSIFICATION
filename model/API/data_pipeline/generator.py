
from typing import List, Dict
import pandas as pd
from .config import PipelineConfig
from .prompt import generate_classification_prompt

class PromptGenerator:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def get_candidates(self, parent_code: str) -> List[Dict]:
        """
        State 5: Candidate Generation
        """
        # Retrieve all valid siblings from HierarchyManager using strict lookup
        candidates = self.config.hierarchy.get_siblings(self.config.level, parent_code)
        
        # Sort alphanumerically by code (as enforced by spec)
        candidates = sorted(candidates, key=lambda x: x['code'])
        return candidates

    def construct_prompt(self, test_sample: pd.Series, support_sample: pd.Series, candidates: List[Dict]) -> str:
        """
        State 6: Prompt Construction
        
        Uses the centralized PromptTemplateManager from prompt.py.
        Note: Model MUST NOT see the DDC codes (blind evaluation).
        """
        # Prepare support example if applicable
        support_dict = None
        support_label = None
        
        if self.config.shot_type != 'zero' and support_sample is not None:
            # Find support sample's label from candidates
            support_code = support_sample.get('ddc_code', '')
            for c in candidates:
                if c['code'] == support_code:
                    support_label = c['label']
                    break
            
            if support_label is None:
                support_label = "Unknown"
            
            support_dict = {
                'title': support_sample.get('title', ''),
                'abstract': support_sample.get('abstract', '')
            }
        
        # Generate prompt using the template manager
        prompt = generate_classification_prompt(
            level=self.config.level,
            candidates=candidates,
            title=test_sample.get('title', ''),
            abstract=test_sample.get('abstract', ''),
            support_sample=support_dict,
            support_label=support_label,
            max_predictions=min(5, len(candidates))
        )
        
        return prompt
