import os
import json
import pandas as pd
import argparse

def process_mitigation_results(base_dir: str, timestamp: str, save_dir: str):
    """
    Parses per-method JSONL files to compute delta metrics and saves statistical summaries.
    Calculates A (initial preference gap) and B (mitigated preference gap).
    """
    os.makedirs(save_dir, exist_ok=True)
    # Define which paraphrase methods were used in the experiment
    methods = ["split", "default", "simplify", "clarify"] 
    
    for method in methods:
        jsonl_path = os.path.join(base_dir, f"paraphrase_{method}_results_{timestamp}.jsonl")
        if not os.path.exists(jsonl_path):
            continue
            
        print(f"Aggregating metrics for: {method}")
        rows = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                model_scores = item.get('model_scores', {})
                
                for model_key, scores in model_scores.items():
                    # Extract raw Log-Likelihoods
                    ll_unf_orig = scores.get('ll_unf_orig', 0)
                    ll_fix_orig = scores.get('ll_fixed_orig', 0)
                    ll_unf_mod = scores.get('ll_unf_mod', 0)
                    ll_fix_mod = scores.get('ll_fixed_mod', 0)

                    # A: Ground truth preference gap in original source
                    # B: Ground truth preference gap in modified source
                    A = ll_fix_orig - ll_unf_orig
                    B = ll_fix_mod - ll_unf_mod
                    
                    rows.append({
                        'model': model_key,
                        'delta_unf': scores.get('delta_unf', 0),
                        'delta_fixed': scores.get('delta_fixed', 0),
                        'A': A,
                        'B': B,
                        'B_minus_A': B - A
                    })

        df = pd.DataFrame(rows)
        summary = []
        for model in df['model'].unique():
            grp = df[df['model'] == model]
            
            # Aggregate stats per model
            summary.append({
                'model': model,
                'mean_delta_unf': grp['delta_unf'].mean(),
                'mean_delta_fix': grp['delta_fixed'].mean(),
                'mean_A': grp['A'].mean(),
                'mean_B': grp['B'].mean(),
                'mean_B_minus_A': grp['B_minus_A'].mean(),
                'A_pos_count': int((grp['A'] > 0).sum()), # Times model preferred 'Fixed' originally
                'B_pos_count': int((grp['B'] > 0).sum()), # Times model preferred 'Fixed' after mitigation
                'sample_size': len(grp)
            })
            
        out_df = pd.DataFrame(summary)
        csv_path = os.path.join(save_dir, f"metrics_{method}.csv")
        out_df.to_csv(csv_path, index=False)
        print(f"  -> Saved {csv_path}")


def save_style_preferences(json_dir: str, timestamp: str, save_dir: str):
    """
    Creates a 'Preference Table' showing whether each model preferred 
    the fixed summary in the original (orig), modified (mod), both, or neither (--) state.
    """
    os.makedirs(save_dir, exist_ok=True)
    styles = ['default', 'split', 'simplify', 'clarify']
    model_map = [
        ('vicuna-7b', 'Vicuna-7B'),
        ('wizardlm-13b', 'WizardLM-13B'),
        ('llama-3.1-8b', 'Llama-3-8B'),
        ('Qwen-3-8b', 'Qwen-3-8B'),
    ]

    for style in styles:
        path = os.path.join(json_dir, f"paraphrase_{style}_results_{timestamp}.jsonl")
        if not os.path.exists(path):
            continue
            
        records = []
        with open(path, 'r') as f:
            for line in f:
                rec = json.loads(line)
                snippet = rec.get('unfaithful_summary', '')[:60]
                row = {'ID': rec.get('document_id'), 'Snippet': snippet}
                
                for key, display_name in model_map:
                    scores = rec['model_scores'].get(key, {})
                    # Did the model correctly prefer the Fixed summary?
                    orig_pref = scores.get('ll_fixed_orig', 0) > scores.get('ll_unf_orig', 0)
                    mod_pref = scores.get('ll_fixed_mod', 0) > scores.get('ll_unf_mod', 0)
                    
                    if orig_pref and mod_pref: label = 'both'
                    elif orig_pref: label = 'orig_only'
                    elif mod_pref: label = 'mod_only'
                    else: label = '--'
                    
                    row[display_name] = label
                records.append(row)

        df = pd.DataFrame(records)
        csv_path = os.path.join(save_dir, f"preferences_{style}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Successfully exported preference table for {style}")


if __name__ == '__main__':
    # Use argparse for better CLI control
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from the experiment logs")
    parser.add_argument("--base_dir", type=str, required=True, help="Directory containing JSONL files")
    args = parser.parse_args()

    # Setup save directory within the base folder
    output_folder = os.path.join(args.base_dir, 'csv_outputs')

    process_mitigation_results(args.base_dir, args.timestamp, output_folder)
    save_style_preferences(args.base_dir, args.timestamp, output_folder)