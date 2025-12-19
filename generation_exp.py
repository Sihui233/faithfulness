import os
import re
import json
import time
import random
import warnings
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict, OrderedDict
from string import Template
from datetime import datetime
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    set_seed,
)
import nltk
from nltk.tokenize import sent_tokenize

# --- External APIs ---
from openai import OpenAI
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# --- Config & Setup ---
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Deterministic mode for reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Download tokenizer data if missing
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# --- Logging & Utils ---

def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{_now()}] {msg}", flush=True)

def _gpu_mem():
    if torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info()
            return f"VRAM: {(total-free)/1e9:.2f}G / {total/1e9:.2f}G"
        except Exception:
            pass
    return "GPU: n/a"

@contextmanager
def stage(name):
    t0 = time.time()
    log(f"START {name} | {_gpu_mem()}")
    try:
        yield
    finally:
        dt = time.time() - t0
        log(f"DONE  {name} in {dt:.2f}s")

def _strip_code_fences(s: str) -> str:
    """Removes markdown code blocks commonly returned by LLMs."""
    if not s: return ""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s).strip()
        s = re.sub(r"\s*```(?:\s*)?$", "", s).strip()
    return s

def _try_extract_first_json_obj(s: str) -> Optional[str]:
    """Robustly finds the first valid JSON bracket pair in mixed text."""
    stack = []
    start = None
    for i, ch in enumerate(s):
        if ch in "{[":
            if not stack: start = i
            stack.append(ch)
        elif ch in "}]":
            if stack:
                opening = stack.pop()
                if (opening, ch) in {("{","}"), ("[","]")} and not stack and start is not None:
                    return s[start:i+1]
    return None

def _normalize_jl(obj, n_doc_sents: int, k_cap: Optional[int]):
    """Standardizes the judge output format."""
    label = (obj.get("label","") if isinstance(obj, dict) else "").strip().lower()
    if label not in ("yes","no"):
        label = "unknown"
    indices = obj.get("indices") if isinstance(obj, dict) else None
    if isinstance(indices, list):
        indices = [int(i) for i in indices if isinstance(i, int) and 0 <= i < n_doc_sents]
        if k_cap and k_cap > 0:
            indices = indices[:k_cap]
    else:
        indices = None
    return {"label": label, "indices": indices}

# --- Data Structures ---

class SimpleSentDoc:
    """Wrapper to handle sentence indexing consistently."""
    def __init__(self, text: str):
        self.original = text
        self.sents = [s.strip() for s in sent_tokenize(text) if s and s.strip()]

    def __len__(self):
        return len(self.sents)

    def get(self, idx: int) -> str:
        return self.sents[idx] if 0 <= idx < len(self.sents) else ""

# --- Models ---

# Configs
GLOBAL_SEED = 42
set_seed(GLOBAL_SEED)

GEN_CFG = dict(
    do_sample=False,    # deterministic greedy/beam
    num_beams=3,
    num_return_sequences=1,
    max_new_tokens=160,
    length_penalty=1.0,
    early_stopping=True,
)

BNB = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_enable_fp32_cpu_offload=True
)

USE_CHAT_TEMPLATE = {
    "vicuna-7b": False,
    "wizardlm-13b": False,
    "llama-3.1-8b": True,
    "Qwen-3-8b": True, 
}

def is_seq2seq_repo(repo_id: str) -> bool:
    repo_id_lower = repo_id.lower()
    return any(k in repo_id_lower for k in ["t5", "flan", "bart", "mbart", "t0"])

def build_summary_prompt(document: str, topic: str) -> str:
    return (
        "### TASK\n"
        f"Summarize the document focusing on '{topic}'. "
        "The summary should be less than 50 words in length."
        "Output ONLY the summary text.\n"
        "### DOCUMENT\n"
        f"{document}\n"
        "### SUMMARY\n"
    )

def extract_after_marker(text: str, marker="SUMMARY") -> str:
    if not text: return ""
    s = text.strip()
    pat = re.compile(rf"(?:^|\n)\s*{re.escape(marker)}\s*[:-]?\s*(?:\n)?", re.IGNORECASE)
    match = None
    for m in pat.finditer(s): match = m
    
    tail = s[match.end():] if match else s
    # Stop at double newline which often indicates end of generation
    return tail.strip().split("\n\n")[0].strip().strip('"\' ')

def post_trim_chat(tok, decoded: str) -> str:
    # Cleanup for chatty models that might repeat headers
    parts = re.split(r"(?:\n\n|^)[Ss]ummary:\s*", decoded)
    return parts[-1] if len(parts) > 1 else decoded

class HFSummarizer:
    def __init__(self, repo_id: str, name: str):
        self.name = name
        self.repo_id = repo_id
        
        log(f"Init {name} ({repo_id})")
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)

        if is_seq2seq_repo(repo_id):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                repo_id, device_map="auto", quantization_config=BNB, low_cpu_mem_usage=True
            )
            self.kind = "seq2seq"
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                repo_id, device_map="auto", quantization_config=BNB, low_cpu_mem_usage=True
            )
            self.kind = "causal"
        self.model.eval()

    def _looks_bad_summary(self, text: str) -> bool:
        t = (text or "").strip().lower()
        return (not t) or (t in {"system", "user", "assistant"}) or (len(t.split()) < 3)

    def _clean_chat_out(self, text: str) -> str:
        s = (text or "").strip().strip('"\' ')
        return re.sub(r'^\s*summary\s*:\s*', '', s, flags=re.I)

    @torch.inference_mode()
    def summarize(self, source: str, topic: str) -> str:
        # 1. Seq2Seq
        if self.kind == "seq2seq":
            prompt = build_summary_prompt(source.strip(), topic.strip())
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)
            gen_ids = self.model.generate(**inputs, **GEN_CFG)
            out = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            return self._clean_chat_out(extract_after_marker(out, "SUMMARY"))

        # 2. Causal (Chat or Plain)
        use_ct = bool(USE_CHAT_TEMPLATE.get(self.name, False) and getattr(self.tokenizer, "apply_chat_template", None))
        
        gen_cfg_local = {**GEN_CFG, "pad_token_id": getattr(self.tokenizer, "eos_token_id", None)}

        if use_ct:
            sys_prompt = f"Summarize the document focusing on '{topic}'. The summary must be < 50 words. Output ONLY the summary text."
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": source.strip()},
                {"role": "assistant", "content": "SUMMARY\n"},
            ]
            batch = self.tokenizer.apply_chat_template(
                messages, tokenize=True, return_tensors="pt", continue_final_message=True
            )
            input_ids = (batch if isinstance(batch, torch.Tensor) else batch["input_ids"]).to(DEVICE)
            
            # Helper for Llama-3 special EOS tokens
            eos_tokens = [self.tokenizer.eos_token_id]
            eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if isinstance(eot, int): eos_tokens.append(eot)
            
            gen_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                eos_token_id=eos_tokens,
                **gen_cfg_local
            )
            out = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            out = post_trim_chat(self.tokenizer, out)
            return self._clean_chat_out(out)

        else:
            # Fallback to plain prompting
            prompt = build_summary_prompt(source.strip(), topic.strip())
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)
            gen_ids = self.model.generate(
                input_ids=inputs["input_ids"], 
                attention_mask=torch.ones_like(inputs["input_ids"]), 
                **gen_cfg_local
            )
            out = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            return self._clean_chat_out(extract_after_marker(out, "### SUMMARY"))

# --- Evaluation ---

class GeminiJudgeAttributor:
    PROMPT = Template(
        """You are an expert annotator performing a combined **faithfulness evaluation** and **attribution** task.
        GOAL
        - Given a source document (as a list of 0-based indexed sentences) and ONE summary sentence, do BOTH:
        1) Attribution: output the **sufficient set** of document sentence indices that together provide coverage for **all factual content** in the summary sentence.
            - "Sufficient" means every claim in the summary is linked to at least one source sentence.
            - Order ascending, no duplicates.
            - If the summary sentence is entirely fabricated or there is no plausible basis, use **None** for "indices".
        2) Faithfulness evaluation: output "yes" if the summary sentence is fully supported by the document; otherwise "no".
            - "Fully supported" means every concrete claim in the summary can be grounded in the document. 
            - Any contradiction, unsupported number/name/date, or missing core fact -> "no".
            

        KEY DEFINITIONS & RULES
        - Never invent indices or text; do not include indices outside the provided range.
        - Decide "yes" when the document allows a reader to verify **all** factual content in the summary.
        - If any part is unverifiable or contradicted, label "no".
        - Numbers, names, dates, counts, and causal attributions must match.
        - Paraphrases are fine if meaning is the same; speculation is not.

        INPUT
        Summary Sentence:
        ${summarySentence}

        Document Sentences (0-based indices):
        ${indexed}

        OUTPUT (STRICT JSON, no extra text, keys exactly as below):
        {
        "indices": [i, j, ...],  // 0-based; [] allowed
                // or None if attribution is None
        "label": "yes" or "no"
        }
    """
    )


    def __init__(self, model="models/gemini-2.0-flash", strict: bool = True):
        self.enabled = HAS_GEMINI
        self.model = model
        self.strict = strict
        if HAS_GEMINI:
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY",""))
            self.client = genai.GenerativeModel(
                self.model,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    response_mime_type="application/json"
                )
            )

    def _indexed_doc(self, source: str):
        doc_sents = [s.strip() for s in sent_tokenize(source) if s.strip()]
        indexed = "\n".join(f"{i}: {s}" for i, s in enumerate(doc_sents))
        return doc_sents, indexed

    def judge_and_link(self, source: str, summary_sentence: str, k_cap: Optional[int] = None):
        if not self.enabled:
            return {"label": "unknown", "indices": None}

        sents, indexed = self._indexed_doc(source)
        prompt = self.PROMPT.substitute(summarySentence=(summary_sentence or "").strip(), indexed=indexed)
        
        try:
            # Rate limit buffer
            time.sleep(1.5)
            out = self.client.generate_content(prompt)
            
            # Extract text safely
            raw = ""
            if hasattr(out, "text"):
                raw = out.text
            else:
                raw = "".join([p.text for p in out.candidates[0].content.parts])

            raw = _strip_code_fences(raw)
            frag = _try_extract_first_json_obj(raw) or raw
            obj = json.loads(frag)

            return _normalize_jl(obj, n_doc_sents=len(sents), k_cap=k_cap)

        except Exception as e:
            # log(f"[Judge Error] {e}")
            return {"label": "unknown", "indices": None}

# --- Modification ---

class SourceModifier:
    """Uses LLM to rewrite specific sentences based on a linguistic strategy."""
    
    self.METHOD_SPECS: Dict[str, str] = {
            # Syntactic (may split/merge if warranted)
            "syntax_transform": (
                "Apply a syntactic transformation that best improves clarity/emphasis while preserving all facts: "
                "split/merge clauses, front/invert a condition/topic, switch active/passive (trim agentless passives), "
                "or integrate appositives via restrictive/nonrestrictive relatives. Use at most one coherent change. "
                "Only split into multiple sentences if splitting is clearly warranted."
            ),
            # Lexical / wording
            "lexical_tighten": (
                "Tighten wording without changing meaning: swap near-synonyms, compress multiword expressions into single strong words "
                'or expand if clearer (e.g., "make a decision" -> "decide"), and replace light-verb constructions with stronger lexical verbs.'
            ),
            # Information structure & discourse
            "discourse_organization": (
                "Improve information structure and discourse flow: order given/background before new/claim, optimize connectors "
                "(avoid stacking; choose however/therefore/moreover/meanwhile appropriately), adjust restrictive vs non-restrictive clauses, "
                "and fix anaphora (disambiguate pronouns or pronominalize repeated NPs)."
            ),
            # Logic templates
            "logic_reframe": (
                "Re-express the logical relation if clearer: causation (X causes Y / Y results from X / due to X), "
                "concession/contrast (Although A, B. / A; however, B.), definition vs property (X is defined as Y <> Y defines X), "
                "or equivalent negation (not uncommon <> common)."
            ),
            # Pragmatics / figurative
            "figurative_normalization": (
                "If the sentence uses figurative/pragmatic devices, normalize them to literal, neutral wording "
                "(e.g., idioms -> literal meaning; sarcasm/irony -> plain statement; hyperbole -> precise/qualified claim; "
                "metaphor -> grounded literal phrase; slang/colloquialisms -> neutral academic register). "
                "Preserve all concrete facts (names/numbers) and add no new information."
            ),
        }

    def __init__(self, model="gpt-4o-mini", temperature=0.2):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    def rewrite(self, sentence: str, method: str) -> Dict[str, Optional[str]]:
        spec = self.METHOD_SPECS.get(method, "Paraphrase preserving facts.")
        prompt = (
            f"OPERATION: {spec}\n"
            "If applicable, output {'can_modify': true, 'rewrite': '...string...'}. "
            "Else {'can_modify': false, 'rewrite': null}. Strict JSON only.\n"
            f"INPUT: {sentence.strip()}"
        )
        
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=256,
            )
            txt = _strip_code_fences(resp.choices[0].message.content)
            frag = _try_extract_first_json_obj(txt) or txt
            obj = json.loads(frag)
            
            can_modify = bool(obj.get("can_modify") is True)
            rewrite = obj.get("rewrite")
            
            if can_modify and isinstance(rewrite, str) and rewrite.strip():
                return {"can_modify": True, "rewrite": rewrite.strip()}
            return {"can_modify": False, "rewrite": None}

        except Exception:
            return {"can_modify": False, "rewrite": None}

# --- Experiment Helpers ---

def apply_rewrites_with_index_map(src_doc: SimpleSentDoc, rewrite_map: Dict[int, str]) -> Tuple[str, Dict[int, List[int]]]:
    """Reconstructs source text and tracks which old index maps to which new indices."""
    new_sents = []
    index_map = {}

    for i, old_sent in enumerate(src_doc.sents):
        if i in rewrite_map and rewrite_map[i]:
            pieces = [t.strip() for t in sent_tokenize(rewrite_map[i]) if t.strip()]
            if not pieces: pieces = [rewrite_map[i].strip()]
            
            start = len(new_sents)
            new_sents.extend(pieces)
            index_map[i] = list(range(start, start + len(pieces)))
        else:
            start = len(new_sents)
            new_sents.append(old_sent)
            index_map[i] = [start]

    return " ".join(new_sents).strip(), index_map

def build_source_edits_log(src_doc, mod_doc, index_map, rewrite_map, method):
    edits = []
    for old_idx, new_idxs in index_map.items():
        if old_idx not in rewrite_map: continue
        edits.append({
            "old_idx": old_idx,
            "method": method,
            "old": src_doc.get(old_idx),
            "new": [mod_doc.get(j) for j in new_idxs],
            "rewrite_raw": rewrite_map[old_idx],
        })
    return edits

def emit_template_record(out_fh, entry, model_name, source_text, summary_text, sent_idx, judge_obj, **kwargs):
    """Writes a standardized JSONL record for analysis."""
    sents = [s.strip() for s in sent_tokenize(summary_text) if s.strip()]
    if not (0 <= sent_idx < len(sents)): return

    rec = {
        **kwargs,
        "dataset": entry.get("dataset", "unknown"),
        "document_id": entry.get("document_id", ""),
        "source": source_text,
        "model": model_name,
        "summary": summary_text,
        "summarySentence": sents[sent_idx],
        "faithfulness_label": judge_obj.get("label"),
        "gemini_annotation": judge_obj.get("indices") or []
    }
    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out_fh.flush()

def safe_judge(judge_linker, source_text, sent, k_cap=3):
    try:
        return judge_linker.judge_and_link(source_text, sent, k_cap=k_cap)
    except Exception:
        return {"label": "unknown", "indices": None}

def _rate(jlist):
    """Calculates unfaithfulness rate: (unfaithful_count, total_count, rate)."""
    valid = [j for j in jlist if j.get("label") in ("yes", "no")]
    if not valid: return (0, 0, 0.0)
    unfaith = sum(1 for j in valid if j["label"] == "no")
    return (unfaith, len(valid), unfaith / len(valid))

# --- Main Runner ---

def run_experiment(data_json: str, output_dir: str, models_to_use: Dict[str,str], methods_to_try: List[str]):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = timestamp
    
    emit_path = os.path.join(output_dir, f"template_stream_{timestamp}.jsonl")
    emit_fh = open(emit_path, "a", encoding="utf-8")

    # Initialize Components
    judge = GeminiJudgeAttributor(strict=True)
    modifier = SourceModifier()
    summarizers = {name: HFSummarizer(repo, name) for name, repo in models_to_use.items()}

    with open(data_json, "r", encoding="utf-8") as f:
        entries = json.load(f)

    # Stats containers
    global_baseline = {m: {"unfaith": 0, "total": 0} for m in summarizers}
    per_model_method = {
        m: {
            mod: {"after_unfaith": 0, "after_total": 0, "not_applicable": defaultdict(int)} 
            for mod in summarizers
        } for m in methods_to_try
    }

    log(f"Starting experiment on {len(entries)} docs...")

    for i_doc, entry in enumerate(entries):
        doc_id = entry.get("document_id", f"doc_{i_doc}")
        source = entry["source"]
        topic = entry.get("topic", "general")
        src_doc = SimpleSentDoc(source)

        for mname, repo in summarizers.items():
            trial_id = f"{doc_id}::{mname}"
            try:
                # --- PHASE 1: Baseline ---
                summary = repo.summarize(source, topic)
                sum_doc = SimpleSentDoc(summary)

                first_pass = []
                for idx, s in enumerate(sum_doc.sents):
                    res = safe_judge(judge, source, s)
                    first_pass.append(res)
                    
                    emit_template_record(
                        emit_fh, entry, mname, source, summary, idx, res,
                        run_id=run_id, phase="first", method="baseline",
                        source_version="original", trial_id=trial_id,
                        pair_key=f"{trial_id}::baseline::{idx}"
                    )

                # Update Baseline Stats
                u1, t1, _ = _rate(first_pass)
                global_baseline[mname]["unfaith"] += u1
                global_baseline[mname]["total"] += t1

                # Identify which source sentences are linked to hallucinations
                bad_idx_with_links = [i for i, r in enumerate(first_pass) if r["label"] == "no" and r["indices"]]
                
                # --- PHASE 2: Mitigation ---
                for method in methods_to_try:
                    
                    # 1. Identify Target Source Indices (Aggregation)
                    to_edit_idxs = sorted({
                        k for bad_idx in bad_idx_with_links
                        for k in (first_pass[bad_idx]["indices"] or [])
                    })

                    if not to_edit_idxs:
                        # Log "No Action" if no source sentences were implicated
                        per_model_method[method][mname]["not_applicable"]["no_link_found"] += 1
                        continue

                    # 2. Modify Source
                    rewrite_map = {}
                    for idx in to_edit_idxs:
                        if 0 <= idx < len(src_doc):
                            res = modifier.rewrite(src_doc.get(idx), method)
                            if res.get("can_modify") and res.get("rewrite"):
                                rewrite_map[idx] = res["rewrite"]

                    if not rewrite_map:
                        # Log "Rewrite Failed"
                        per_model_method[method][mname]["not_applicable"]["rewrite_failed"] += 1
                        continue

                    # 3. Apply Edits & Re-Summarize
                    mod_source, index_map = apply_rewrites_with_index_map(src_doc, rewrite_map)
                    mod_doc = SimpleSentDoc(mod_source)
                    source_edits = build_source_edits_log(src_doc, mod_doc, index_map, rewrite_map, method)

                    re_summary = repo.summarize(mod_source, topic)
                    re_sum_doc = SimpleSentDoc(re_summary)

                    # 4. Re-Judge (Second Pass)
                    second_pass = []
                    for idx, s in enumerate(re_sum_doc.sents):
                        res = safe_judge(judge, mod_source, s)
                        second_pass.append(res)

                        emit_template_record(
                            emit_fh, entry, mname, mod_source, re_summary, idx, res,
                            run_id=run_id, phase="second", method=method,
                            source_version="modified", trial_id=trial_id,
                            pair_key=f"{trial_id}::{method}::all::{idx}",
                            source_edits=source_edits
                        )

                    # Update Mitigation Stats
                    u2, t2, _ = _rate(second_pass)
                    per_model_method[method][mname]["after_unfaith"] += u2
                    per_model_method[method][mname]["after_total"] += t2

            except Exception as e:
                log(f"[ERROR] {doc_id} / {mname}: {e}")
                continue

    # Final Aggregation
    summary_path = os.path.join(output_dir, f"aggregate_{timestamp}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"baseline": global_baseline, "methods": per_model_method}, f, indent=2)

    emit_fh.close()
    log(f"Done. Results in {output_dir}")

# --- Deduplication Helper (Optional) ---

def dedupe_and_merge(in_path: str, out_path: str):
    """Cleans dataset based on DocID+Topic keys."""
    with open(in_path, "r") as f:
        rows = json.load(f)

    groups = OrderedDict()
    for r in rows:
        key = f"{r.get('document_id')}:::{r.get('topic','').strip()}"
        groups.setdefault(key, []).append(r)

    cleaned = []
    for items in groups.values():
        # Heuristic: keep the one with the longest source text
        best = max(items, key=lambda x: len(x.get("source","")))
        cleaned.append(best)

    with open(out_path, "w") as f:
        json.dump(cleaned, f, indent=2)
    return out_path

# --- Execution ---

if __name__ == "__main__":
    # Example Configuration
    INPUT_JSON = "~/data/uniq_doc_topic_with_meta.json"   
    OUT_DIR = os.path.expanduser(f"~/scratch/direct_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    MODELS = {
        "vicuna-7b": "lmsys/vicuna-7b-v1.5",
        "wizardlm-13b": "WizardLMTeam/WizardLM-13B-V1.2",
        "llama-3.1-8b":"meta-llama/Llama-3.1-8B-Instruct",
        "Qwen-3-8b":"Qwen/Qwen3-8B"
    }
    
    METHODS = ["syntax_transform", "discourse_organization"]
    
    run_experiment(INPUT_JSON, OUT_DIR, MODELS, METHODS)