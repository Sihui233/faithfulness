import os
import json
import time
import gc
import warnings
import random
import argparse
from datetime import datetime
from typing import Optional, List, Dict

import torch
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

# Hugging Face & LLM imports
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    logging as hf_logging,
)
from openai import OpenAI

# --- Setup & Config ---

# Mute the noise from HF libraries
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Scoring models
HF_LLM_MODELS = {
    "vicuna-7b": "lmsys/vicuna-7b-v1.5",
    "wizardlm-13b": "WizardLMTeam/WizardLM-13B-V1.2",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen-3-8b": "Qwen/Qwen3-8B"
}

def set_seed(seed: int = 42):
    """Reproducibility matters."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Classes ---

class HFScorer:
    """
    Wraps a Hugging Face Causal LM to calculate conditional log-likelihoods.
    """
    def __init__(self, model_name: str):
        print(f"Loading scorer: {model_name}...")
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        # Load in 8-bit to save VRAM (crucial for running multiple 13B+ models)
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,             
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=True      
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config, 
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def conditional_loglikelihood(self, source: str, whole_summ: str, target_sent: str, topic: str) -> float:
        """
        Calculates P(target_sent | source, summary_prefix).
        """
        # 1. Prompt Construction
        sys_prompt = (
            f"You are a helpful assistant. Summarize the provided document faithfully and concisely on the {topic}. "
            "The summary should not contain any information not supported by the document. " 
            "The summary should be less than 50 words in length."
        )

        # Find where the target sentence sits in the summary to build the prefix
        summary_sentences = sent_tokenize(whole_summ)
        
        # Fuzzy matching index in case of whitespace diffs
        target_idx = -1
        for i, s in enumerate(summary_sentences):
            if s.strip() == target_sent.strip():
                target_idx = i
                break
        
        if target_idx == -1:
            # Fallback: if tokenization splits differently, assume it's the whole string or handle error
            # For this script, we assume strict matching logic from data prep
            print(f"Warning: Target sentence not found in summary. returning default -inf.")
            return -999.0

        prefix = "\n".join(summary_sentences[:target_idx])
        target = target_sent.strip()

        # 2. Tokenization
        if getattr(self.tok, "chat_template", None):
            # Modern chat models (Llama-3, Qwen) handle this structure natively
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": source.strip()},
                {"role": "assistant", "content": prefix.strip()},
            ]
            batch = self.tok.apply_chat_template(
                messages,
                tokenize=True,
                continue_final_message=True, # crucial: lets us append the target token next
                return_tensors="pt"
            )
            input_ids = batch.to(DEVICE) if isinstance(batch, torch.Tensor) else batch["input_ids"].to(DEVICE)
        else:
            prompt_str = (
                f"{sys_prompt}\n\nDocument:\n{source.strip()}\n\n"
                f"Summary so far:\n{prefix.strip()}\nAnswer:"
            )
            input_ids = self.tok(prompt_str, return_tensors="pt")["input_ids"].to(DEVICE)

        target_ids = self.tok(target, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)

        # 3. Concatenate: [Context] + [Target]
        full_input = torch.cat([input_ids, target_ids], dim=1)
        
        # 4. Masking
        # Only want to calculate loss on the *target* tokens, not the context
        labels = full_input.clone()
        labels[:, :input_ids.shape[1]] = -100 

        # 5. Inference
        with torch.inference_mode():
            out = self.model(input_ids=full_input, labels=labels)
            # Loss is average NLL, so -loss is average LogLikelihood
            avg_logp = -out.loss.item()

        # Cleanup VRAM immediately
        del input_ids, target_ids, full_input, labels, out
        torch.cuda.empty_cache()

        return avg_logp


class GPTParaphraser:
    """
    Uses OpenAI API to rewrite sentences using specific strategies (simplification, splitting, etc.).
    """
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        self.client = OpenAI() # expects OPENAI_API_KEY in env
        self.model = model
        self.temperature = temperature
        self.system_prompt = (
            "You are a helpful assistant that paraphrases and simplifies sentences "
            "without changing meaning or omitting any details (including named entities and numbers)."
        )

    def paraphrase(self, text: str, method: str = "default") -> str:
        """
        Dispatches to different prompt strategies based on 'method'.
        """
        prompts = {
            "simplify": (
                "Simplify the following sentence without removing any important detail, including names and numbers. "
                "Avoid rephrasing it into multiple sentences. Do not add, assume, or invent any new information.\n"
                "Return only the final rewritten sentence."
            ),
            "split": (
                "Rephrase and break the following sentence into two or more shorter, simpler sentences. "
                "Keep all key facts, named entities, and numbers. Do not add, assume, or invent any new information.\n"
                "Return only the final rewritten sentence."
            ),
            "clarify": (
                "Rewrite the sentence to make it clearer and easier to understand, but do not change its meaning. "
                "Do not add, assume, or invent any new information.\n"
                "Return only the final rewritten sentence."
            ),
            "default": (
                "Paraphrase and simplify the following sentence without changing its meaning, "
                "and without omitting or altering important details.\n"
                "Return only the final rewritten sentence."
            )
        }

        user_prompt = f"{prompts.get(method, prompts['default'])}\n\n{text.strip()}"

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=200, 
                n=1,
            )
            return resp.choices[0].message.content.strip().strip('"\'')
        except Exception as e:
            print(f"[API ERROR] {e}")
            return text # fallback to original if API fails

# --- Data Helpers ---

def load_data(json_path: str, split: str = "dev"):
    """
    Filters the dataset for 'unfaithful' examples that have a 'fixed' counterpart.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Only care about unfaithful examples where we know the 'fix' (ground truth)
    # and ignoring 'extrinsic' hallucinations which are hard to fix by just changing source style.
    filtered = [
        entry for entry in data 
        if entry.get("faithfulness_label", "").lower() == "no"
        and entry.get("split", "").lower() == split
        and entry.get("type", "").lower() != "extrinsic information"
        and entry.get("fixedSummarySentence") is not None
    ]
    return filtered

# --- Main Pipeline ---

def run_experiment(data_path: str, output_dir: str):
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Data
    unfaithful_data = load_data(data_path)
    print(f"Loaded {len(unfaithful_data)} unfaithful examples.")

    # Initialize Models
    scorers = {name: HFScorer(repo) for name, repo in HF_LLM_MODELS.items()}
    paraphraser = GPTParaphraser()
    
    methods = ["simplify", "split", "clarify", "default"]

    for method in methods:
        print(f"\n===== Method: {method} =====\n")
        
        res_file = os.path.join(output_dir, f"paraphrase_{method}_{timestamp}.jsonl")
        delta_file = os.path.join(output_dir, f"delta_{method}_summary_{timestamp}.json")
        
        delta_logs = {name: [] for name in scorers}

        for i, entry in enumerate(unfaithful_data):
            doc_id = entry.get("document_id", "N/A")
            print(f"[{i+1}/{len(unfaithful_data)}] Processing {doc_id}...")

            # 1. Paraphrase the source sentences identified by annotation
            original_source = entry["source"]
            target_sentences = entry.get("gemini_annotation", [])
            
            rewrite_map = {}
            modified_source = original_source

            for sent in target_sentences:
                rewritten = paraphraser.paraphrase(sent, method=method)
                if rewritten == sent:
                    print(f"  [Warn] Paraphrase identical for: {sent[:30]}...")
                rewrite_map[sent] = rewritten
                
                # Naive string replacement, assuming sentence doesn't appear twice, 
                # sufficient for this experimental setup.
                modified_source = modified_source.replace(sent, rewritten)

            # 2. Score with LLMs
            # See if Modifying the source makes the 'Unfaithful' summary LESS likely
            # and the 'Fixed' summary MORE likely.
            
            summ_full = entry["summary"]
            unfaith_sent = entry["summarySentence"]
            fixed_sent = entry["fixedSummarySentence"]
            topic = entry.get("topic", "general topic")

            model_scores = {}

            for model_name, scorer in scorers.items():
                # Score Unfaithful Summary
                ll_unf_orig = scorer.conditional_loglikelihood(original_source, summ_full, unfaith_sent, topic)
                ll_unf_mod = scorer.conditional_loglikelihood(modified_source, summ_full, unfaith_sent, topic)
                
                # Score Fixed Summary
                ll_fix_orig = scorer.conditional_loglikelihood(original_source, summ_full, fixed_sent, topic)
                ll_fix_mod = scorer.conditional_loglikelihood(modified_source, summ_full, fixed_sent, topic)

                # Delta > 0 means the likelihood INCREASED after modification
                # Delta < 0 means the likelihood DECREASED after modification
                delta_unf = ll_unf_mod - ll_unf_orig
                delta_fix = ll_fix_mod - ll_fix_orig

                delta_logs[model_name].append({
                    "delta_unf": delta_unf,
                    "delta_fixed": delta_fix
                })

                model_scores[model_name] = {
                    "ll_unf_orig": ll_unf_orig,
                    "ll_unf_mod": ll_unf_mod,
                    "delta_unf": delta_unf,
                    "ll_fix_orig": ll_fix_orig,
                    "ll_fix_mod": ll_fix_mod,
                    "delta_fixed": delta_fix
                }

            # 3. Save individual result
            result = {
                "document_id": doc_id,
                "method": method,
                "rewrite_map": rewrite_map,
                "scores": model_scores
            }
            
            with open(res_file, "a") as f:
                f.write(json.dumps(result) + "\n")

        # 4. Summarize Method Results
        print(f"\n--- Summary for {method} ---")
        summary_stats = {}
        for name, records in delta_logs.items():
            if not records: continue
            
            unf_deltas = [r["delta_unf"] for r in records]
            fix_deltas = [r["delta_fixed"] for r in records]
            
            avg_unf = sum(unf_deltas) / len(unf_deltas)
            avg_fix = sum(fix_deltas) / len(fix_deltas)
            
            summary_stats[name] = {
                "avg_delta_unf": avg_unf,
                "avg_delta_fix": avg_fix,
                "count": len(records)
            }
            print(f"{name}: Unfaithful Δ {avg_unf:.4f} | Fixed Δ {avg_fix:.4f}")

        with open(delta_file, "w") as f:
            json.dump(summary_stats, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/unfaithful_w_fixed_gem_annot.json", help="Input JSON file")
    parser.add_argument("--out_dir", type=str, default="~/scratch/mitigation_fix", help="Output directory")
    args = parser.parse_args()

    set_seed(42)
    
    # Expand user path if ~ is used
    out_dir_expanded = os.path.expanduser(args.out_dir)
    
    print(f"System Check | Device: {DEVICE}")
    run_experiment(args.data, out_dir_expanded)