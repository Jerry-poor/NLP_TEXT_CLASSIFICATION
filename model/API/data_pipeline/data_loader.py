
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .config import PipelineConfig, RANDOM_SEED

class DataLoader:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def load_and_clean(self) -> pd.DataFrame:
        """
        State 2: Data Loading & Cleaning
        """
        # 1. Read CSV
        df = pd.read_csv(self.config.dataset_path, dtype=str)
        
        # 2. Standardize columns
        cols = df.columns.str.lower()
        rename_map = {}
        
        if 'title' in cols:
            rename_map[df.columns[cols.tolist().index('title')]] = 'title'
        if 'abstract' in cols:
            rename_map[df.columns[cols.tolist().index('abstract')]] = 'abstract'
            
        # DDC Code mapping based on Level
        # Try finding specific column for this level first
        target_col = f"ddc_l{self.config.level}" 
        possible_code_cols = [target_col, 'ddc_code', 'ddc']
        found_code_col = None
        
        # Case insensitive search
        df_cols_lower = [c.lower() for c in df.columns]
        for p in possible_code_cols:
            if p.lower() in df_cols_lower:
                idx = df_cols_lower.index(p.lower())
                found_code_col = df.columns[idx]
                break
        
        if found_code_col:
            rename_map[found_code_col] = 'ddc_code'
        else:
            # If explicit L{x} column missing, warn or fail?
            # For now try to use 'ddc_code' generic if exists
            raise ValueError(f"Could not find DDC code column (e.g., {target_col}) in {self.config.dataset_path}")

        df = df.rename(columns=rename_map)
        
        # Fill NA
        df['title'] = df.get('title', pd.Series([''] * len(df))).fillna('')
        df['abstract'] = df.get('abstract', pd.Series([''] * len(df))).fillna('')
        df['ddc_code'] = df['ddc_code'].fillna('').astype(str).str.strip().str.zfill(3)

        # 3. Filter rows where ddc_code is valid using HierarchyManager
        initial_len = len(df)
        
        # Vectorized check is hard with custom logic, use apply
        def check_validity(code):
            return self.config.hierarchy.is_valid(self.config.level, code)
            
        mask = df['ddc_code'].apply(check_validity)
        df = df[mask].copy()
        
        print(f"Loaded {len(df)}/{initial_len} valid samples for Level {self.config.level}")
        
        # 4. Derive parent code using HierarchyManager lookup (No manual slicing!)
        def lookup_parent(code):
            return self.config.hierarchy.get_parent(self.config.level, code)
            
        df['parent_code'] = df['ddc_code'].apply(lookup_parent)
        
        # Drop rows where parent could not be found (should be implicitly handled by validity, but safe check)
        df = df.dropna(subset=['parent_code'])
        
        return df

    def select_support_set(self, df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], List[int]]:
        """
        State 3: Support Set Selection (Deterministic)
        """
        support_map = {}
        used_indices = []
        
        if self.config.shot_type == 'zero':
            return support_map, used_indices

        # Group by parent_code (Bucket)
        groups = df.groupby('parent_code')
        for parent, group in groups:
            # Sort by index to be deterministic
            sorted_group = group.sort_index()
            if not sorted_group.empty:
                # Pick first available sample
                support_sample = sorted_group.iloc[0]
                support_map[parent] = support_sample
                used_indices.append(support_sample.name)
        
        return support_map, used_indices

    def sample_test_set(self, df: pd.DataFrame, used_indices: List[int]) -> pd.DataFrame:
        """
        State 4: Test Set Sampling
        """
        # 1. Exclude used
        df_clean = df.drop(used_indices, errors='ignore')
        
        # 2. Sample N rows using Fixed Seed
        # Use simple random sample with seed
        if len(df_clean) > self.config.sample_size:
            test_df = df_clean.sample(n=self.config.sample_size, random_state=RANDOM_SEED)
        else:
            test_df = df_clean
            
        return test_df
