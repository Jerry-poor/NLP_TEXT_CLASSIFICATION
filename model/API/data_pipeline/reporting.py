
import os
import pandas as pd
from typing import Dict
from .config import PipelineConfig

class Reporter:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def generate_report(self, metrics: Dict, results_df: pd.DataFrame):
        """
        State 10: Reporting
        """
        # Ensure output directory exists (with model subdirectory)
        model_name = self.config.deepseek_model if self.config.deepseek_model else "default"
        safe_model_name = "".join([c for c in model_name if c.isalnum() or c in ('-', '_')]).strip()
        save_dir = os.path.join(self.config.output_dir, safe_model_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare content
        report_content = []
        report_content.append(f"# Evaluation Report")
        report_content.append(f"**Level**: {self.config.level}")
        report_content.append(f"**Dataset**: {os.path.basename(self.config.dataset_path)}")
        report_content.append(f"**Shot Type**: {self.config.shot_type}")
        report_content.append(f"**Sample Size**: {metrics.get('total_samples', len(results_df))}")
        report_content.append(f"**Model**: {self.config.deepseek_model}")
        report_content.append("")
        
        report_content.append("## Accuracy Metrics")
        report_content.append(f"| Metric | Value | Count |")
        report_content.append(f"|--------|-------|-------|")
        report_content.append(f"| **Top-1** | {metrics.get('top1_accuracy', 0):.2%} | {metrics.get('top1_correct', 0)}/{metrics.get('total_samples', 0)} |")
        report_content.append(f"| **Top-3** | {metrics.get('top3_accuracy', 0):.2%} | {metrics.get('top3_correct', 0)}/{metrics.get('total_samples', 0)} |")
        report_content.append(f"| **Top-5** | {metrics.get('top5_accuracy', 0):.2%} | {metrics.get('top5_correct', 0)}/{metrics.get('total_samples', 0)} |")
        report_content.append("")
        
        # Weighted Metrics Section
        report_content.append("## Weighted Metrics (Top-1)")
        report_content.append(f"| Metric | Value |")
        report_content.append(f"|--------|-------|")
        report_content.append(f"| **Accuracy** | {metrics.get('accuracy', 0):.4f} |")
        report_content.append(f"| **Precision (Weighted)** | {metrics.get('weighted_precision', 0):.4f} |")
        report_content.append(f"| **Recall (Weighted)** | {metrics.get('weighted_recall', 0):.4f} |")
        report_content.append(f"| **F1 Score (Weighted)** | {metrics.get('weighted_f1', 0):.4f} |")
        report_content.append("")
        
        report_content.append("## Latency")
        report_content.append(f"- **Avg Duration**: {metrics.get('avg_duration_ms', 0):.2f} ms")
        report_content.append(f"- **Total Duration**: {metrics.get('total_duration_s', 0):.2f} s")
        report_content.append("")
        
        report_content.append("## Sample Results (First 10)")
        # Show first 10 matches/mismatches
        display_cols = ['ddc_code', 'predicted_codes', 'title']
        available_cols = [c for c in display_cols if c in results_df.columns]
        if available_cols:
            sample_view = results_df[available_cols].head(10).copy()
            # Ensure ddc_code is 3-digit string for display
            if 'ddc_code' in sample_view.columns:
                sample_view['ddc_code'] = sample_view['ddc_code'].astype(str).apply(lambda x: x.zfill(3))
            report_content.append(sample_view.to_markdown(index=False))
        
        # File Name Construction
        dataset_name = os.path.splitext(os.path.basename(self.config.dataset_path))[0]
        filename = f"{dataset_name}_level{self.config.level}_sample{metrics.get('total_samples', 0)}_{self.config.shot_type}_{self.config.deepseek_model}.md"
        output_path = os.path.join(save_dir, filename)
        
        with open(output_path, "w", encoding='utf-8') as f:
            f.write("\n".join(report_content))
            
        # Also save CSV results
        csv_filename = filename.replace('.md', '.csv')
        results_df.to_csv(os.path.join(save_dir, csv_filename), index=False)
        
        print(f"Report saved to {output_path}")
