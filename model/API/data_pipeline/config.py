
import os
import argparse
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

# Fixed Constants from Spec
RANDOM_SEED = 42
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 200
WORKERS = 50
MAX_RETRIES = 3
TIMEOUT = 60
FREQ_PENALTY = 0.0
PRES_PENALTY = 0.0

# =============================================================================
# PROVIDER CONFIGURATION
# =============================================================================

PROVIDERS = {
    'deepseek': {
        'api_key_env': 'DEEPSEEK_API_KEY',
        'base_url': 'https://api.deepseek.com',  # Fixed, not from env
        'default_model': 'deepseek-chat',
        'model_prefixes': ['deepseek'],
    },
    'openai': {
        'api_key_env': 'OPENAI_API_KEY',
        'base_url': 'https://api.openai.com/v1',  # Fixed, not from env
        'default_model': 'gpt-4o-mini',
        'model_prefixes': ['gpt-', 'o1', 'o3'],
    },
    'qwen': {
        'api_key_env': 'QWEN_API_KEY',
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',  # Fixed
        'default_model': 'qwen-plus',
        'model_prefixes': ['qwen'],
    },
    'siliconflow': {
        'api_key_env': 'SILICONFLOW_API_KEY',
        'base_url': 'https://api.siliconflow.cn/v1',  # Fixed
        'default_model': 'deepseek-ai/DeepSeek-V3',
        'model_prefixes': [],  # Uses org/model format detection
    },
}

def get_provider_from_model(model_name: str) -> str:
    """
    Infer provider from model name prefix or pattern.
    """
    model_lower = model_name.lower()
    
    # Check each provider's prefixes
    for provider_name, config in PROVIDERS.items():
        for prefix in config.get('model_prefixes', []):
            if model_lower.startswith(prefix):
                return provider_name
    
    # Special case: org/model format -> SiliconFlow
    if '/' in model_name:
        return 'siliconflow'
    
    # Default to deepseek
    return 'deepseek'


@dataclass
class PipelineConfig:
    level: int
    sample_size: int
    dataset_path: str
    shot_type: str  # 'zero', 'one', 'few'
    output_dir: str
    api_key: str
    base_url: str
    model: str
    provider: str
    mapping_df: pd.DataFrame
    hierarchy: Any # HierarchyManager instance
    
    # Aliases for backward compatibility
    @property
    def deepseek_api_key(self):
        return self.api_key
    
    @property
    def deepseek_base_url(self):
        return self.base_url
    
    @property
    def deepseek_model(self):
        return self.model

class HierarchyManager:
    def __init__(self, mapping_df: pd.DataFrame):
        self.df = mapping_df
        self.parent_lookup = {} # (level, code) -> parent_code_str
        self.buckets = {}       # (level, parent_code_str) -> List[Dict] {code, label, definition}
        self.details = {}       # (level, code) -> Dict
        self._build_hierarchy()

    def _build_hierarchy(self):
        # Sets of valid codes per level
        valid_codes = {1: set(), 2: set(), 3: set()}
        
        # 1. Parse all rows to populate valid codes and details
        for _, row in self.df.iterrows():
            try:
                # Parse
                lvl_str = str(row['Level']).strip()
                if not lvl_str.isdigit(): continue
                lvl = int(lvl_str)
                
                code = str(row['Sequence']).strip().zfill(3)
                label = str(row['Label']).strip()
                defn = str(row['definition']).strip()
                
                if lvl not in [1, 2, 3]: continue
                
                valid_codes[lvl].add(code)
                self.details[(lvl, code)] = {
                    'code': code,
                    'label': label,
                    'definition': defn
                }
            except Exception as e:
                # skip malformed
                continue

        # 2. Build Links (Parent -> Children Buckets) using DDC Prefix Logic
        
        # Level 1: Parent is 'root'
        self.buckets[(1, 'root')] = []
        sorted_l1 = sorted(list(valid_codes[1]))
        for code in sorted_l1:
            self.parent_lookup[(1, code)] = 'root'
            self.buckets[(1, 'root')].append(self.details[(1, code)])

        # Level 2: Parent is X00
        sorted_l2 = sorted(list(valid_codes[2]))
        for code in sorted_l2:
            parent_code = f"{code[0]}00"
            if parent_code in valid_codes[1]:
                self.parent_lookup[(2, code)] = parent_code
                bucket_key = (2, parent_code)
                if bucket_key not in self.buckets:
                    self.buckets[bucket_key] = []
                self.buckets[bucket_key].append(self.details[(2, code)])

        # Level 3: Parent is XX0
        sorted_l3 = sorted(list(valid_codes[3]))
        for code in sorted_l3:
            parent_code = f"{code[:2]}0"
            if parent_code in valid_codes[2]:
                self.parent_lookup[(3, code)] = parent_code
                bucket_key = (3, parent_code)
                if bucket_key not in self.buckets:
                    self.buckets[bucket_key] = []
                self.buckets[bucket_key].append(self.details[(3, code)])
    
    def get_parent(self, level: int, code: str) -> Optional[str]:
        return self.parent_lookup.get((level, code))

    def get_siblings(self, level: int, parent_code: str) -> List[Dict]:
        return self.buckets.get((level, parent_code), [])

    def is_valid(self, level: int, code: str) -> bool:
        return (level, code) in self.details

class ConfigLoader:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.env_path = os.path.join(base_dir, '.env')
        self.mapping_path = os.path.join(base_dir, 'DDClabel_deepseek_hierarchical.csv')
        self.datasets_dir = os.path.join(base_dir, 'datasets')
        self.results_dir = os.path.join(base_dir, 'result')

        load_dotenv(self.env_path)

    def load_mapping_manager(self) -> Tuple[pd.DataFrame, HierarchyManager]:
        if not os.path.exists(self.mapping_path):
            raise FileNotFoundError(f"Mapping file not found at {self.mapping_path}")
        
        df = pd.read_csv(self.mapping_path, dtype=str)
        manager = HierarchyManager(df)
        return df, manager

    def parse_args(self) -> PipelineConfig:
        parser = argparse.ArgumentParser(description="Deterministic Hierarchical Evaluation Pipeline")
        parser.add_argument('--level', type=int, choices=[1, 2, 3], required=True, help='DDC Level')
        parser.add_argument('--sample_size', type=int, required=True, help='Test samples count')
        parser.add_argument('--dataset', type=str, required=True, help='Dataset filename')
        parser.add_argument('--shot_type', type=str, choices=['zero', 'one', 'few'], required=True)
        
        # Model configuration
        parser.add_argument('--model', type=str, required=False, help='Model name (auto-detects provider)')
        parser.add_argument('--provider', type=str, choices=list(PROVIDERS.keys()), required=False,
                          help='Explicitly specify provider (deepseek, openai, qwen, siliconflow)')
        parser.add_argument('--base_url', type=str, required=False, help='Override base URL (optional)')
        parser.add_argument('--api_key', type=str, required=False, help='Override API key (optional)')
        
        args = parser.parse_args()

        dataset_full_path = os.path.join(self.datasets_dir, args.dataset)
        if not os.path.exists(dataset_full_path):
            raise FileNotFoundError(f"Dataset {dataset_full_path} not found")

        mapping_df, hierarchy = self.load_mapping_manager()
        
        # =================================================================
        # PROVIDER AND MODEL RESOLUTION
        # =================================================================
        
        # Step 1: Determine provider
        if args.provider:
            # Explicitly specified
            provider = args.provider
        elif args.model:
            # Auto-detect from model name
            provider = get_provider_from_model(args.model)
        else:
            # Default provider
            provider = 'deepseek'
        
        provider_config = PROVIDERS[provider]
        
        # Step 2: Get model name (CLI > Env > default for provider)
        env_model_key = f"{provider.upper()}_MODEL"
        model_name = args.model or os.getenv(env_model_key) or provider_config['default_model']
        
        # Step 3: Get base URL (CLI override > fixed provider URL)
        # NOTE: We use the FIXED base_url from provider config, not from env
        # This prevents misconfiguration issues
        base_url = args.base_url or provider_config['base_url']
        
        # Step 4: Get API key (CLI > env)
        api_key = args.api_key or os.getenv(provider_config['api_key_env'])
        
        if not api_key:
            raise ValueError(
                f"API key not found for provider '{provider}'.\n"
                f"Please set {provider_config['api_key_env']} in your .env file."
            )
        
        print(f"[Config] Provider: {provider}")
        print(f"[Config] Model: {model_name}")
        print(f"[Config] Base URL: {base_url}")

        return PipelineConfig(
            level=args.level,
            sample_size=args.sample_size,
            dataset_path=dataset_full_path,
            shot_type=args.shot_type,
            output_dir=self.results_dir,
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            provider=provider,
            mapping_df=mapping_df,
            hierarchy=hierarchy
        )
