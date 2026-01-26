"""
Prompt Template Manager for DDC Classification Pipeline.

This module provides a centralized template for generating classification prompts,
supporting dynamic substitution of categories (based on DDC level) and text content.
"""

from typing import List, Dict

# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

PROMPT_TEMPLATE = """You are a Dewey Decimal {level_description} classifier.

Your task is to classify the given Title and Abstract into the MOST appropriate categories from the allowed list.

IMPORTANT PRINCIPLE (MANDATORY):
You must perform DEFINITION-DRIVEN classification.
All decisions MUST be based ONLY on semantic alignment between the text and the provided category descriptions.
Do NOT rely on prior knowledge, topic popularity, typical venues, or common classifications.
Do NOT infer missing details.

Allowed categories (output MUST match these names EXACTLY):

{categories_block}

Decision procedure (MANDATORY):
1. Compare the Title and Abstract against EACH category description.
2. Select the categories whose descriptions can explain the text with the FEWEST additional assumptions.
3. Rank categories by degree of semantic coverage.
4. IMPORTANT: You MUST return EXACTLY {max_predictions} predictions (unless the total number of allowed categories is less than {max_predictions}, in which case return all of them).
5. If only 1-2 categories are strong matches, you MUST fill the remaining slots with the next most plausible categories, even if their confidence is low (e.g., 0.05). Do NOT return fewer than {max_predictions} items.

Confidence assignment rules (MANDATORY — this is rule 4):
- Confidence represents SEMANTIC EXCLUSIVITY, not model certainty.
- If two or more categories plausibly explain the text, the top confidence MUST be reduced.
- In ambiguous or interdisciplinary cases, the highest confidence MUST NOT exceed 0.75.
- The sum of confidence scores does not need to equal 1.0.
- Use low confidence scores (e.g., 0.05 - 0.15) for forced filling items.

Output rules:
1) Output MUST be a single valid JSON object. No extra text. No markdown.
2) Return EXACTLY {max_predictions} predictions (unless total categories < {max_predictions}).
3) Use EXACT category names from the allowed list.
4) JSON schema:
{{
  "predictions": [
    {{"category": "<exact allowed name>", "confidence": <float 0.00 to 1.00>}},
    ...
  ]
}}
5) confidence MUST be a float in [0, 1] with up to 2 decimal places.

{shot_block}Now classify the following text STRICTLY using the rules above:

Title: {title}
Abstract: {abstract}
"""

# =============================================================================
# SHOT TEMPLATE (for one-shot / few-shot learning)
# =============================================================================

SHOT_TEMPLATE = """Support example:
Title: {shot_title}
Abstract: {shot_abstract}
Correct Label: ### {shot_label} ###

"""


class PromptTemplateManager:
    """
    Manages prompt generation for DDC classification tasks.
    
    Supports:
    - Dynamic category substitution based on DDC level
    - Zero-shot, one-shot, and few-shot configurations
    - Consistent formatting across the pipeline
    """
    
    def __init__(self):
        self.template = PROMPT_TEMPLATE
        self.shot_template = SHOT_TEMPLATE
    
    def get_level_description(self, level: int) -> str:
        """
        Returns a human-readable description of the DDC level.
        """
        level_map = {
            1: "top-level",
            2: "second-level",
            3: "third-level"
        }
        return level_map.get(level, f"level-{level}")
    
    def format_categories(self, candidates: List[Dict]) -> str:
        """
        Formats candidate categories into the prompt block.
        """
        lines = []
        for cand in candidates:
            label = cand.get('label', '')
            definition = cand.get('definition', '')
            
            if definition:
                # Multi-line format: label on first line, definition indented
                lines.append(f"- {label}:")
                lines.append(f"  {definition}")
                lines.append("")  # Empty line for readability
            else:
                lines.append(f"- {label}")
                lines.append("")
        
        return "\n".join(lines).strip()
    
    def format_shot(self, support_sample: dict, support_label: str) -> str:
        """
        Formats a support example for one-shot/few-shot learning.
        """
        if support_sample is None:
            return ""
        
        return self.shot_template.format(
            shot_title=support_sample.get('title', ''),
            shot_abstract=support_sample.get('abstract', ''),
            shot_label=support_label
        )
    
    def generate_prompt(
        self,
        level: int,
        candidates: List[Dict],
        title: str,
        abstract: str,
        support_sample: dict = None,
        support_label: str = None,
        max_predictions: int = 5
    ) -> str:
        """
        Generates a complete classification prompt.
        """
        # Get level description
        level_description = self.get_level_description(level)
        
        # Format categories block
        categories_block = self.format_categories(candidates)
        
        # Determine max predictions (capped by number of candidates)
        num_candidates = len(candidates)
        effective_max = min(max_predictions, num_candidates)
        
        # Format shot block
        shot_block = ""
        if support_sample is not None and support_label is not None:
            shot_block = self.format_shot(support_sample, support_label)
        
        # Generate final prompt
        prompt = self.template.format(
            level_description=level_description,
            categories_block=categories_block,
            max_predictions=effective_max,
            shot_block=shot_block,
            title=title,
            abstract=abstract
        )
        
        return prompt


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

# Global instance for easy import
prompt_manager = PromptTemplateManager()


def generate_classification_prompt(
    level: int,
    candidates: List[Dict],
    title: str,
    abstract: str,
    support_sample: dict = None,
    support_label: str = None,
    max_predictions: int = 5
) -> str:
    """
    Convenience function to generate a classification prompt.
    """
    return prompt_manager.generate_prompt(
        level=level,
        candidates=candidates,
        title=title,
        abstract=abstract,
        support_sample=support_sample,
        support_label=support_label,
        max_predictions=max_predictions
    )
