import os
import json
import pandas as pd


def process_mitigation_results(base_dir: str, timestamp: str, save_dir: str):
    """
    Read per-method JSONL results, compute metrics, and save summary CSVs.
    """
    os.makedirs(save_dir, exist_ok=True)
    methods = ["split", "default"] # "simplify", "clarify",
    jsonl_paths = {
        name: os.path.join(base_dir, f"paraphrase_{name}_results_{timestamp}.jsonl")
        for name in methods
    }

    for method, jsonl_path in jsonl_paths.items():
        print(f"Processing {method}: {jsonl_path}")
        rows = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                model_scores = item.get('model_scores', {})
                for model_key, scores in model_scores.items():
                    ll_unf_orig = scores.get('ll_unf_orig', 0)
                    ll_fix_orig = scores.get('ll_fixed_orig', 0)
                    ll_unf_mod = scores.get('ll_unf_mod', 0)
                    ll_fix_mod = scores.get('ll_fixed_mod', 0)

                    delta_unf = scores.get('delta_unf', None)
                    delta_fixed = scores.get('delta_fixed', None)

                    A = ll_fix_orig - ll_unf_orig
                    B = ll_fix_mod - ll_unf_mod
                    B_minus_A = B - A

                    rows.append({
                        'model': model_key,
                        'delta_unf': delta_unf,
                        'delta_fixed': delta_fixed,
                        'A': A,
                        'B': B,
                        'B_minus_A': B_minus_A
                    })

        df = pd.DataFrame(rows)
        summary = []
        for model in df['model'].unique():
            grp = df[df['model'] == model]
            A_vals = grp['A']
            B_vals = grp['B']

            summary.append({
                'model': model,
                'mean_delta_unf': grp['delta_unf'].mean(),
                'mean_delta_fix': grp['delta_fixed'].mean(),
                'mean_A': A_vals.mean(),
                'mean_B': B_vals.mean(),
                'mean_B_minus_A': grp['B_minus_A'].mean(),
                'A_pos_count': int((A_vals > 0).sum()),
                'A_neg_count': int((A_vals < 0).sum()),
                'B_pos_count': int((B_vals > 0).sum()),
                'B_neg_count': int((B_vals < 0).sum()),
                'n': len(grp)
            })
        out_df = pd.DataFrame(summary)
        csv_path = os.path.join(save_dir, f"results_{method}.csv")
        out_df.to_csv(csv_path, index=False)
        print(f"Saved summary CSV to {csv_path}")


def save_style_preferences(json_dir: str, timestamp: str, save_dir: str):
    """
    Load JSONL paraphrase results and save preference tables as CSV for each style.
    """
    os.makedirs(save_dir, exist_ok=True)
    styles = ['default', 'split'] #, 'simplify', 'clarify'
    model_map = [
        ('vicuna-7b', 'Vicuna-7B'),
        ('wizardlm-13b', 'WizardLM-13B'),
        ('llama-3.1-8b', 'Llama-3-8B'),
        ('Qwen-3-8b', 'Qwen-3-8B'),
    ]

    for style in styles:
        records = []
        path = os.path.join(json_dir, f"paraphrase_{style}_results_{timestamp}.jsonl")
        with open(path, 'r') as f:
            for line in f:
                rec = json.loads(line)
                snippet = rec.get('unfaithful_summary', '')[:60] + '…'
                row = {'ID': rec.get('document_id'), 'Unf snippet': snippet}
                for key, name in model_map:
                    scores = rec['model_scores'].get(key, {})
                    orig_pref = scores.get('ll_fixed_orig', 0) > scores.get('ll_unf_orig', 0)
                    mod_pref = scores.get('ll_fixed_mod', 0) > scores.get('ll_unf_mod', 0)
                    if orig_pref and mod_pref:
                        label = 'both'
                    elif orig_pref:
                        label = 'orig'
                    elif mod_pref:
                        label = 'mod'
                    else:
                        label = '--'
                    row[name + ' Pref'] = label
                records.append(row)

        df = pd.DataFrame(records)
        csv_path = os.path.join(save_dir, f"preferences_{style}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved preferences CSV for {style} to {csv_path}")


if __name__ == '__main__':
    TIMESTAMP = "20250811_154747"  # update as needed
    BASE_DIR = f"path_to_file_{TIMESTAMP}"
    SAVE_DIR = os.path.join(BASE_DIR, 'csv_outputs')  # folder for all CSV exports

    # Process and save mitigation summaries
    process_mitigation_results(BASE_DIR, TIMESTAMP, SAVE_DIR)
    # Save preference tables without display
    save_style_preferences(BASE_DIR, TIMESTAMP, SAVE_DIR)
