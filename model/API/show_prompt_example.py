
import pandas as pd
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from data_pipeline.generator import PromptGenerator

# Mock Config mimicking the pipeline configuration
class MockConfig:
    def __init__(self):
        self.level = 1
        self.shot_type = 'one' # Showing one-shot example to demonstrate the full prompt structure
        self.hierarchy = MockHierarchy()

class MockHierarchy:
    def get_siblings(self, level, code):
        # Mocking a subset of DDC Level 1 categories
        return [
            {'code': '000', 'label': 'Computer science, information & general works', 'definition': 'The interdisciplinary field encompassing the theoretical foundations of information and computation.'},
            {'code': '500', 'label': 'Science', 'definition': 'The comprehensive systematic enterprise that builds and organizes knowledge through testable explanations and predictions about the natural world.'},
            {'code': '600', 'label': 'Technology', 'definition': 'General works on technology and applied sciences that treat the subject broadly.'},
            {'code': '300', 'label': 'Social sciences', 'definition': 'The interdisciplinary field of academic study that systematically investigates human societies.'},
            {'code': '100', 'label': 'Philosophy & psychology', 'definition': 'The systematic rational inquiry into fundamental questions concerning existence, knowledge, values, reason, mind, and language.'}
        ]

def main():
    config = MockConfig()
    generator = PromptGenerator(config)
    
    # Mock Test Sample (The text to be classified)
    test_sample = pd.Series({
        'title': 'Deep Learning for Image Recognition in Autonomous Vehicles',
        'abstract': 'This study proposes a novel convolutional neural network architecture for real-time object detection in self-driving cars. We evaluate the performance...',
        'parent_code': 'root',
        'ddc_code': '000' # Ground truth (hidden from model in real pipeline)
    })
    
    # Mock Support Sample (The example provided in the prompt)
    support_sample = pd.Series({
        'title': 'Quantum Mechanics and the structure of the atom',
        'abstract': 'An overview of the historical development of atomic theory, focusing on the contributions of Bohr, Schrodinger, and Heisenberg.',
        'ddc_code': '500' # Corresponds to Science
    })
    
    # Get candidates (mocked)
    candidates = config.hierarchy.get_siblings(1, 'root')
    
    # Construct the prompt
    prompt = generator.construct_prompt(test_sample, support_sample, candidates)
    
    print(prompt)

if __name__ == "__main__":
    main()
