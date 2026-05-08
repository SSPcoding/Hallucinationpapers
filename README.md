FYRP Progress Review–1
Detailed Content Guide

1. Problem Statement & Motivation
Must include:
Clear, 2–3 line problem definition (not generic)
Domain/context (e.g., healthcare, IoT, NLP)
Real-world relevance (who benefits, how)
Avoid: vague goals like “improve accuracy using AI”

2. Objectives of the Study
Must include:
1 primary objective (core goal)
2–3 specific measurable objectives
e.g., “compare ResNet50 vs EfficientNet on MRI dataset”
Avoid: mixing objectives with methodology

3. Literature Review & Research Gap
Literature Review must include:
Minimum 10 recent and relevant papers 
Table or bullets with:
Method used
Dataset
Key result
Research Gap must include:
2–3 clear limitations from literature
Logical link to your proposed work

4. Proposed Methodology / Approach
Must include:
✔ Pipeline (stepwise flow)
Input → preprocessing → model → output
Model choices (e.g., CNN, transfer learning)
Any improvement idea (attention, ensemble, etc.)
✔ Schematic Model Representation 
Students must include a clear schematic diagram of the proposed system
Diagram should show:
Input (e.g., image, text, sensor data)
Preprocessing steps
Model architecture
Enhancement techniques (if any)
Output layer
Diagram should be original and well-labelled (not copied blindly)

5. Dataset & Tools/Technologies
Dataset Description:
Source (Kaggle, hospital, open dataset, etc.)
Size, classes, imbalance (if any)
Sample images (if applicable)
Tools must include:
Language (Python, etc.)
Frameworks (TensorFlow, PyTorch etc.)
Hardware (GPU/CPU/cloud etc.)

6. Work Done So Far
Must include:
Literature survey completion status
Dataset collection/preprocessing done
Model(s) implemented (even basic)
Should reflect actual progress, not plans

7. Preliminary Results (if any)
Must include (if available):
Accuracy / loss
Sample outputs / predictions
Observations (1–2 insights)
If not available: explicitly state and justify

8. Future Work Plan
Must include:
Specific next steps:
Model tuning
Trying new architectures
Evaluation metrics
Clear technical direction
Avoid: generic “we will improve the model”

9. References
Must include:
Minimum 10 properly formatted citations (IEEE format only)
Only relevant papers, not blogs/random links

# Stage 1
# ============================================================
# STAGE 1 FINAL:
# Load PopQA, filter to clean relations, build incomplete KG
#
# Output files:
# - data_final/stage1_popqa_clean_relations.csv
# - data_final/kg_full_clean.csv
# - data_final/kg_incomplete_clean.csv
# - data_final/kg_removed_missing_true_clean.csv
# - data_final/stage1_questions_clean_with_kg_labels.csv
# - data_final/stage1_clean_metadata.json
# ============================================================

!pip install -q datasets pandas numpy tqdm

import os
import json
import random
import numpy as np
import pandas as pd

from datasets import load_dataset, get_dataset_split_names

# -----------------------------
# Config
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = "data_final"
os.makedirs(DATA_DIR, exist_ok=True)

DATASET_NAME = "akariasai/PopQA"

# Clean relations for final paper
CLEAN_RELATIONS = [
    "author",
    "composer",
    "director",
    "producer",
    "screenwriter",
    "capital",
    "place of birth",
    "country",
    "developer",
    "publisher"
]

# Use full filtered set first.
# Later, for faster experiments, you can reduce this.
SAMPLE_SIZE = 2500

# Create incomplete KG by removing this fraction of true triples.
MISSING_FRACTION = 0.45

# If True, use only low-popularity / long-tail subject entities.
# For first final run, keep False so we still have enough data.
LONG_TAIL_ONLY = False

print("Clean relations:", CLEAN_RELATIONS)


# ============================================================
# 1. Load PopQA
# ============================================================

split_names = get_dataset_split_names(DATASET_NAME)
print("Available splits:", split_names)

split_to_use = "test" if "test" in split_names else split_names[0]
print("Using split:", split_to_use)

ds = load_dataset(DATASET_NAME, split=split_to_use)
df = ds.to_pandas()

print("Raw PopQA rows:", len(df))
print("Columns:", list(df.columns))


# ============================================================
# 2. Basic cleaning
# ============================================================

required_cols = [
    "id",
    "subj", "prop", "obj",
    "subj_id", "prop_id", "obj_id",
    "question", "possible_answers",
    "s_wiki_title", "o_wiki_title",
    "s_pop", "o_pop"
]

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing expected columns: {missing_cols}")

df = df.dropna(subset=["subj", "prop", "obj", "question"]).copy()

for col in ["subj", "prop", "obj", "subj_id", "prop_id", "obj_id", "question"]:
    df[col] = df[col].astype(str).str.strip()

df["prop_normalized"] = df["prop"].astype(str).str.lower().str.strip()

df["s_pop"] = pd.to_numeric(df["s_pop"], errors="coerce").fillna(0)
df["o_pop"] = pd.to_numeric(df["o_pop"], errors="coerce").fillna(0)

df["triple_id"] = (
    df["subj_id"].astype(str) + "||" +
    df["prop_id"].astype(str) + "||" +
    df["obj_id"].astype(str)
)

df = df.drop_duplicates(subset=["triple_id"]).reset_index(drop=True)

print("Rows after basic cleaning:", len(df))


# ============================================================
# 3. Inspect relation availability
# ============================================================

print("\nTop PopQA relations:")
display(df["prop_normalized"].value_counts().head(30).reset_index().rename(
    columns={"index": "relation", "prop_normalized": "count"}
))

clean_relation_set = set([r.lower().strip() for r in CLEAN_RELATIONS])

filtered_df = df[df["prop_normalized"].isin(clean_relation_set)].copy().reset_index(drop=True)

print("\nRows after clean-relation filtering:", len(filtered_df))
print("Relations kept:")
display(filtered_df["prop_normalized"].value_counts().reset_index().rename(
    columns={"index": "relation", "prop_normalized": "count"}
))

if len(filtered_df) == 0:
    raise ValueError(
        "No rows matched the clean relations. Inspect df['prop'].unique() and adjust CLEAN_RELATIONS."
    )


# ============================================================
# 4. Popularity bins for long-tail analysis
# ============================================================

def make_popularity_bins(series):
    """
    Creates low / medium / high popularity bins using ranked quantiles.
    This is safer when many popularity values are repeated.
    """
    try:
        return pd.qcut(
            series.rank(method="first"),
            q=3,
            labels=["low_popularity", "medium_popularity", "high_popularity"]
        )
    except Exception:
        return pd.Series(["unknown_popularity"] * len(series), index=series.index)

filtered_df["subject_popularity_bin"] = make_popularity_bins(filtered_df["s_pop"])
filtered_df["object_popularity_bin"] = make_popularity_bins(filtered_df["o_pop"])

print("\nSubject popularity bins:")
print(filtered_df["subject_popularity_bin"].value_counts())

if LONG_TAIL_ONLY:
    filtered_df = filtered_df[
        filtered_df["subject_popularity_bin"] == "low_popularity"
    ].copy().reset_index(drop=True)
    print("\nUsing long-tail only.")
    print("Rows after long-tail filtering:", len(filtered_df))


# ============================================================
# 5. Create a balanced sample across relations and popularity
# ============================================================

def balanced_sample_by_relation_and_popularity(input_df, sample_size, seed=42):
    """
    Samples roughly across relation + subject popularity bin groups.
    This prevents one relation from dominating the dataset.
    """
    if sample_size is None or sample_size >= len(input_df):
        return input_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    input_df = input_df.copy()
    input_df["sample_group"] = (
        input_df["prop_normalized"].astype(str) + "||" +
        input_df["subject_popularity_bin"].astype(str)
    )

    groups = list(input_df.groupby("sample_group"))
    n_groups = len(groups)

    base_per_group = max(1, sample_size // n_groups)

    sampled_parts = []

    for group_name, group in groups:
        n_take = min(base_per_group, len(group))
        sampled_parts.append(group.sample(n=n_take, random_state=seed))

    sample_df = pd.concat(sampled_parts).drop_duplicates(subset=["triple_id"])

    # Top up if needed
    if len(sample_df) < sample_size:
        remaining = input_df[~input_df["triple_id"].isin(sample_df["triple_id"])]
        needed = sample_size - len(sample_df)
        if len(remaining) > 0:
            topup = remaining.sample(n=min(needed, len(remaining)), random_state=seed)
            sample_df = pd.concat([sample_df, topup])

    # Trim if needed
    if len(sample_df) > sample_size:
        sample_df = sample_df.sample(n=sample_size, random_state=seed)

    sample_df = sample_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if "sample_group" in sample_df.columns:
        sample_df = sample_df.drop(columns=["sample_group"])

    return sample_df

sample_df = balanced_sample_by_relation_and_popularity(
    filtered_df,
    SAMPLE_SIZE,
    seed=SEED
)

print("\nFinal Stage 1 sample rows:", len(sample_df))
print("Unique relations:", sample_df["prop_normalized"].nunique())

print("\nSample relation distribution:")
display(sample_df["prop_normalized"].value_counts().reset_index().rename(
    columns={"index": "relation", "prop_normalized": "count"}
))

print("\nSample subject popularity distribution:")
print(sample_df["subject_popularity_bin"].value_counts())


# ============================================================
# 6. Build full local KG
# ============================================================

kg_full = sample_df[[
    "triple_id",
    "subj_id", "subj",
    "prop_id", "prop",
    "prop_normalized",
    "obj_id", "obj",
    "s_pop", "o_pop",
    "subject_popularity_bin",
    "object_popularity_bin",
    "s_wiki_title",
    "o_wiki_title"
]].copy()

kg_full = kg_full.rename(columns={
    "subj_id": "subject_id",
    "subj": "subject_label",
    "prop_id": "relation_id",
    "prop": "relation_label",
    "prop_normalized": "relation_normalized",
    "obj_id": "object_id",
    "obj": "object_label"
})

print("\nFull clean KG triples:", len(kg_full))
display(kg_full.head())


# ============================================================
# 7. Create incomplete KG
# ============================================================

sample_df["removal_group"] = (
    sample_df["prop_normalized"].astype(str) + "||" +
    sample_df["subject_popularity_bin"].astype(str)
)

removed_parts = []

for group_name, group in sample_df.groupby("removal_group"):
    n_remove = int(round(len(group) * MISSING_FRACTION))

    # Avoid very tiny groups being wiped out.
    if len(group) >= 3 and n_remove >= 1:
        removed_parts.append(group.sample(n=n_remove, random_state=SEED))

if len(removed_parts) > 0:
    removed_df = pd.concat(removed_parts).drop_duplicates(subset=["triple_id"]).reset_index(drop=True)
else:
    removed_df = pd.DataFrame(columns=sample_df.columns)

kept_df = sample_df[
    ~sample_df["triple_id"].isin(removed_df["triple_id"])
].copy().reset_index(drop=True)

kg_removed = kg_full[
    kg_full["triple_id"].isin(removed_df["triple_id"])
].copy().reset_index(drop=True)

kg_incomplete = kg_full[
    kg_full["triple_id"].isin(kept_df["triple_id"])
].copy().reset_index(drop=True)

actual_missing_fraction = len(kg_removed) / len(kg_full) if len(kg_full) > 0 else 0

print("\nOriginal clean KG triples:", len(kg_full))
print("Incomplete clean KG triples:", len(kg_incomplete))
print("Removed true triples:", len(kg_removed))
print("Actual missing fraction:", round(actual_missing_fraction, 4))


# ============================================================
# 8. Build labelled question file
# ============================================================

stage1_questions = sample_df.copy()

stage1_questions["gold_kg_label"] = np.where(
    stage1_questions["triple_id"].isin(kg_incomplete["triple_id"]),
    "KG_SUPPORTED",
    "MISSING_BUT_TRUE"
)

stage1_questions = stage1_questions[[
    "id",
    "question",
    "subj_id", "subj",
    "prop_id", "prop", "prop_normalized",
    "obj_id", "obj",
    "possible_answers",
    "s_wiki_title", "o_wiki_title",
    "s_pop", "o_pop",
    "subject_popularity_bin",
    "object_popularity_bin",
    "triple_id",
    "gold_kg_label"
]].copy()

stage1_questions = stage1_questions.rename(columns={
    "subj_id": "subject_id",
    "subj": "subject_label",
    "prop_id": "relation_id",
    "prop": "relation_label",
    "prop_normalized": "relation_normalized",
    "obj_id": "gold_object_id",
    "obj": "gold_object_label"
})

print("\nGold KG label distribution:")
print(stage1_questions["gold_kg_label"].value_counts())

print("\nGold KG label by relation:")
display(pd.crosstab(
    stage1_questions["relation_normalized"],
    stage1_questions["gold_kg_label"]
))


# ============================================================
# 9. Save outputs
# ============================================================

sample_df.to_csv(f"{DATA_DIR}/stage1_popqa_clean_relations.csv", index=False)
kg_full.to_csv(f"{DATA_DIR}/kg_full_clean.csv", index=False)
kg_incomplete.to_csv(f"{DATA_DIR}/kg_incomplete_clean.csv", index=False)
kg_removed.to_csv(f"{DATA_DIR}/kg_removed_missing_true_clean.csv", index=False)
stage1_questions.to_csv(f"{DATA_DIR}/stage1_questions_clean_with_kg_labels.csv", index=False)

metadata = {
    "dataset": DATASET_NAME,
    "split_used": split_to_use,
    "random_seed": SEED,
    "clean_relations": CLEAN_RELATIONS,
    "long_tail_only": LONG_TAIL_ONLY,
    "raw_popqa_rows": int(len(df)),
    "rows_after_clean_relation_filtering": int(len(filtered_df)),
    "sample_size": int(len(sample_df)),
    "full_kg_triples": int(len(kg_full)),
    "incomplete_kg_triples": int(len(kg_incomplete)),
    "removed_true_triples": int(len(kg_removed)),
    "missing_fraction_requested": float(MISSING_FRACTION),
    "missing_fraction_actual": float(actual_missing_fraction),
    "relation_distribution": sample_df["prop_normalized"].value_counts().to_dict(),
    "subject_popularity_distribution": sample_df["subject_popularity_bin"].value_counts().astype(int).to_dict(),
    "labels": {
        "KG_SUPPORTED": "Gold triple exists in the incomplete KG.",
        "MISSING_BUT_TRUE": "Gold triple was removed from the KG but remains a true fact."
    },
    "output_files": {
        "stage1_sample": f"{DATA_DIR}/stage1_popqa_clean_relations.csv",
        "kg_full": f"{DATA_DIR}/kg_full_clean.csv",
        "kg_incomplete": f"{DATA_DIR}/kg_incomplete_clean.csv",
        "kg_removed": f"{DATA_DIR}/kg_removed_missing_true_clean.csv",
        "questions_with_labels": f"{DATA_DIR}/stage1_questions_clean_with_kg_labels.csv"
    }
}

with open(f"{DATA_DIR}/stage1_clean_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved Stage 1 final files:")
for file in os.listdir(DATA_DIR):
    print("-", file)

print("\nMetadata:")
print(json.dumps(metadata, indent=2))


# ============================================================
# 10. Quick examples
# ============================================================

print("\nExample KG-supported questions:")
display(stage1_questions[stage1_questions["gold_kg_label"] == "KG_SUPPORTED"][[
    "question",
    "subject_label",
    "relation_label",
    "gold_object_label",
    "subject_popularity_bin",
    "gold_kg_label"
]].head(5))

print("\nExample missing-but-true questions:")
display(stage1_questions[stage1_questions["gold_kg_label"] == "MISSING_BUT_TRUE"][[
    "question",
    "subject_label",
    "relation_label",
    "gold_object_label",
    "subject_popularity_bin",
    "gold_kg_label"
]].head(5))
# Stage 2
# ============================================================
# STAGE 2 FINAL REFINED:
# Generate GenAI answers using Qwen2.5-3B-Instruct
# and evaluate answers using PopQA possible_answers only.
#
# Input:
# - data_final/stage1_questions_clean_with_kg_labels.csv
#
# Output:
# - data_final/stage2_model_answers.csv
# - data_final/stage2_model_answer_metadata.json
# - data_final/stage2_popqa_alias_audit.csv
#
# Core design:
# - Raw model answers are preserved.
# - No manual alias dictionary.
# - No Wikidata alias fetching.
# - Matching uses only:
#   1. main PopQA reference answer
#   2. PopQA possible_answers
# - Final labels use human-friendly names for paper readability.
# ============================================================

!pip install -q transformers accelerate bitsandbytes sentencepiece pandas numpy tqdm

import os
import re
import ast
import json
import time
import random
import unicodedata
import numpy as np
import pandas as pd
import torch

from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ============================================================
# 0. Configuration
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = "data_final"
os.makedirs(DATA_DIR, exist_ok=True)

STAGE1_PATH = f"{DATA_DIR}/stage1_questions_clean_with_kg_labels.csv"

OUTPUT_PATH = f"{DATA_DIR}/stage2_model_answers.csv"
METADATA_PATH = f"{DATA_DIR}/stage2_model_answer_metadata.json"
ALIAS_AUDIT_PATH = f"{DATA_DIR}/stage2_popqa_alias_audit.csv"

# If you want a larger run later, increase this.
NUM_ROWS_FOR_STAGE2 = 500

GENAI_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# If True, always generate new Qwen answers.
# If False, code will reuse existing raw answers if available.
FORCE_REGENERATE_MODEL_ANSWERS = False

# If an older Stage 2 file exists, this allows reusing its raw answers.
# This does not reuse old labels.
POSSIBLE_OLD_STAGE2_FILES = [
    f"{DATA_DIR}/stage2_model_answers.csv",
    f"{DATA_DIR}/stage2_genai_answers_clean.csv"
]

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: No GPU detected. Qwen2.5-3B will be very slow on CPU.")
    print("Recommended: Runtime -> Change runtime type -> T4 GPU")


# ============================================================
# 1. Human-friendly label constants
# ============================================================

KG_FACT_STATUS = {
    "PRESENT": "FACT_PRESENT_IN_KG",
    "MISSING": "FACT_MISSING_FROM_KG",
    "UNKNOWN": "UNKNOWN_KG_STATUS"
}

ANSWER_CASE = {
    "CORRECT_PRESENT": "CORRECT_AND_FACT_PRESENT_IN_KG",
    "CORRECT_MISSING": "CORRECT_BUT_FACT_MISSING_FROM_KG",
    "WRONG": "WRONG_MODEL_ANSWER",
    "UNUSABLE": "UNUSABLE_MODEL_ANSWER"
}

ANSWER_MATCH_TYPE = {
    "EXACT_REFERENCE": "EXACT_REFERENCE_ANSWER",
    "POPQA_ALIAS": "POPQA_ALIAS",
    "NO_MATCH": "NO_MATCH",
    "INVALID": "INVALID_OUTPUT"
}

ANSWER_MATCH_SOURCE = {
    "REFERENCE": "REFERENCE_ANSWER",
    "POPQA": "POPQA_POSSIBLE_ANSWER",
    "NO_MATCH": "NO_MATCH",
    "INVALID": "INVALID_OUTPUT"
}

STAGE1_KG_LABEL_TO_NEW_STATUS = {
    "KG_SUPPORTED": KG_FACT_STATUS["PRESENT"],
    "MISSING_BUT_TRUE": KG_FACT_STATUS["MISSING"]
}


# ============================================================
# 2. Load Stage 1
# ============================================================

if not os.path.exists(STAGE1_PATH):
    raise FileNotFoundError(
        f"Stage 1 file not found: {STAGE1_PATH}\n"
        "Run Stage 1 first."
    )

stage1_df = pd.read_csv(STAGE1_PATH)

print("\nLoaded Stage 1 rows:", len(stage1_df))
print("Columns:", list(stage1_df.columns))

required_cols = [
    "id",
    "question",
    "subject_id",
    "subject_label",
    "relation_id",
    "relation_label",
    "relation_normalized",
    "gold_object_id",
    "gold_object_label",
    "possible_answers",
    "s_wiki_title",
    "o_wiki_title",
    "subject_popularity_bin",
    "triple_id",
    "gold_kg_label"
]

missing_cols = [c for c in required_cols if c not in stage1_df.columns]
if missing_cols:
    raise ValueError(f"Missing required Stage 1 columns: {missing_cols}")


# Rename Stage 1 columns into reviewer-friendly Stage 2 names.
questions_df = stage1_df.rename(columns={
    "gold_object_id": "reference_answer_id",
    "gold_object_label": "reference_answer",
    "possible_answers": "popqa_possible_answers"
}).copy()

questions_df["kg_fact_status"] = (
    questions_df["gold_kg_label"]
    .map(STAGE1_KG_LABEL_TO_NEW_STATUS)
    .fillna(KG_FACT_STATUS["UNKNOWN"])
)

# Remove old Stage 1 label from Stage 2 working data.
questions_df = questions_df.drop(columns=["gold_kg_label"], errors="ignore")

print("\nKG fact status distribution:")
print(questions_df["kg_fact_status"].value_counts())

print("\nRelation distribution:")
print(questions_df["relation_normalized"].value_counts())

print("\nPopularity distribution:")
print(questions_df["subject_popularity_bin"].value_counts())


# ============================================================
# 3. Balanced Stage 2 sampling
# ============================================================

def balanced_sample(input_df, sample_size, seed=42):
    """
    Balanced across:
    - KG fact status
    - relation
    - subject popularity bin
    """
    if sample_size is None or sample_size >= len(input_df):
        return input_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    df = input_df.copy()

    df["sample_group"] = (
        df["kg_fact_status"].astype(str) + "||" +
        df["relation_normalized"].astype(str) + "||" +
        df["subject_popularity_bin"].astype(str)
    )

    groups = list(df.groupby("sample_group"))
    base_per_group = max(1, sample_size // len(groups))

    sampled_parts = []

    for _, group in groups:
        n_take = min(base_per_group, len(group))
        sampled_parts.append(group.sample(n=n_take, random_state=seed))

    sample_df = pd.concat(sampled_parts).drop_duplicates(subset=["triple_id"])

    if len(sample_df) < sample_size:
        remaining = df[~df["triple_id"].isin(sample_df["triple_id"])]
        needed = sample_size - len(sample_df)

        if len(remaining) > 0:
            topup = remaining.sample(
                n=min(needed, len(remaining)),
                random_state=seed
            )
            sample_df = pd.concat([sample_df, topup])

    if len(sample_df) > sample_size:
        sample_df = sample_df.sample(n=sample_size, random_state=seed)

    sample_df = sample_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    sample_df = sample_df.drop(columns=["sample_group"], errors="ignore")

    return sample_df


stage2_df = balanced_sample(
    questions_df,
    NUM_ROWS_FOR_STAGE2,
    seed=SEED
)

print("\nStage 2 selected rows:", len(stage2_df))

print("\nStage 2 KG fact status distribution:")
print(stage2_df["kg_fact_status"].value_counts())

print("\nStage 2 relation distribution:")
print(stage2_df["relation_normalized"].value_counts())

print("\nStage 2 popularity distribution:")
print(stage2_df["subject_popularity_bin"].value_counts())


# ============================================================
# 4. Try reusing existing raw model answers
# ============================================================

stage2_df["model_raw_answer"] = np.nan

if not FORCE_REGENERATE_MODEL_ANSWERS:
    reused_file = None

    for candidate_path in POSSIBLE_OLD_STAGE2_FILES:
        if os.path.exists(candidate_path):
            try:
                old_df = pd.read_csv(candidate_path)

                raw_col = None
                if "model_raw_answer" in old_df.columns:
                    raw_col = "model_raw_answer"
                elif "genai_raw_answer" in old_df.columns:
                    raw_col = "genai_raw_answer"

                if raw_col is not None and "triple_id" in old_df.columns:
                    raw_answers = (
                        old_df[["triple_id", raw_col]]
                        .dropna(subset=[raw_col])
                        .drop_duplicates(subset=["triple_id"])
                        .rename(columns={raw_col: "reused_model_raw_answer"})
                    )

                    before_missing = stage2_df["model_raw_answer"].isna().sum()

                    stage2_df = stage2_df.merge(
                        raw_answers,
                        on="triple_id",
                        how="left"
                    )

                    stage2_df["model_raw_answer"] = stage2_df["model_raw_answer"].fillna(
                        stage2_df["reused_model_raw_answer"]
                    )

                    stage2_df = stage2_df.drop(columns=["reused_model_raw_answer"])

                    after_missing = stage2_df["model_raw_answer"].isna().sum()

                    if after_missing < before_missing:
                        reused_file = candidate_path
                        print(f"\nReused raw model answers from: {candidate_path}")
                        print("Reused answers:", before_missing - after_missing)

            except Exception as e:
                print(f"Could not reuse answers from {candidate_path}: {e}")

    if reused_file is None:
        print("\nNo reusable raw model answers found. Will generate answers.")


missing_answer_mask = (
    stage2_df["model_raw_answer"].isna()
    | (stage2_df["model_raw_answer"].astype(str).str.strip() == "")
)

num_missing_answers = int(missing_answer_mask.sum())

print("\nRows needing model generation:", num_missing_answers)


# ============================================================
# 5. Generate Qwen answers if needed
# ============================================================

elapsed_seconds = 0.0
batch_size_used = None
num_generated_now = 0

if num_missing_answers > 0:

    tokenizer = AutoTokenizer.from_pretrained(
        GENAI_MODEL_NAME,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model = AutoModelForCausalLM.from_pretrained(
            GENAI_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            GENAI_MODEL_NAME,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model.to(device)

    model.eval()

    print("\nLoaded GenAI model:", GENAI_MODEL_NAME)

    def build_prompt(question):
        return f"""Answer the question with only the shortest possible entity name.
Do not explain.
Do not write a full sentence.
Do not include punctuation unless it is part of the name.

Question: {question}
Answer:"""

    def generate_answers_batch(questions, batch_size=4, max_new_tokens=24):
        all_answers = []

        for start in tqdm(range(0, len(questions), batch_size)):
            batch_questions = questions[start:start + batch_size]
            prompts = [build_prompt(q) for q in batch_questions]

            messages_batch = [
                [{"role": "user", "content": prompt}]
                for prompt in prompts
            ]

            texts = [
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                for messages in messages_batch
            ]

            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]
            decoded = tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )

            cleaned = [ans.strip() for ans in decoded]
            all_answers.extend(cleaned)

        return all_answers

    batch_size_used = 4 if device == "cuda" else 1

    missing_indices = stage2_df.index[missing_answer_mask].tolist()
    missing_questions = stage2_df.loc[missing_indices, "question"].tolist()

    start_time = time.time()

    generated_answers = generate_answers_batch(
        missing_questions,
        batch_size=batch_size_used,
        max_new_tokens=24
    )

    elapsed_seconds = time.time() - start_time
    num_generated_now = len(generated_answers)

    for idx, answer in zip(missing_indices, generated_answers):
        stage2_df.at[idx, "model_raw_answer"] = answer

    print("\nGenerated answers now:", num_generated_now)
    print("Elapsed seconds:", round(elapsed_seconds, 2))

else:
    print("\nAll raw model answers were reused. No model generation needed.")


# ============================================================
# 6. Text normalization utilities
# ============================================================

def strip_accents(text):
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def normalize_text(text):
    """
    Conservative normalization for answer matching.
    No manual alias mapping is used.
    """
    if pd.isna(text):
        return ""

    text = str(text).strip()
    text = strip_accents(text)
    text = text.lower()

    text = text.replace("_", " ")
    text = text.replace("-", " ")

    text = re.sub(r"[\“\”\‘\’\"']", "", text)
    text = re.sub(r"\([^)]*\)", " ", text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def compact_text(text):
    """
    Compact form catches formatting differences:
    - WashingtonDC vs Washington D C
    - Qasr-e-Shirin vs Qasr e Shirin
    """
    return re.sub(r"[^a-z0-9]", "", normalize_text(text))


def answer_forms(text):
    """
    Returns two answer forms:
    1. normalized text with spaces
    2. compact text without spaces
    """
    forms = set()

    if pd.isna(text):
        return forms

    raw = str(text).strip()

    if not raw:
        return forms

    norm = normalize_text(raw)
    compact = compact_text(raw)

    if norm:
        forms.add(norm)

    if compact:
        forms.add(compact)

    return forms


# ============================================================
# 7. Parse PopQA possible_answers
# ============================================================

def parse_popqa_possible_answers(value):
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, list):
            return [str(x) for x in parsed if str(x).strip()]

        if isinstance(parsed, tuple):
            return [str(x) for x in parsed if str(x).strip()]

        return [str(parsed)]

    except Exception:
        text = text.strip("[]")
        parts = re.split(r",|\|", text)

        return [
            p.strip().strip("'").strip('"')
            for p in parts
            if p.strip()
        ]


# ============================================================
# 8. Extract short entity answer from raw model output
# ============================================================

def extract_entity_answer(raw_answer, subject_label=None):
    """
    Extract a short entity-like answer from the raw model output.

    Important:
    - This does not manually correct the model.
    - This does not remove hallucinated names.
    - If the model gives a weird entity, we keep it as a wrong candidate.
    """
    if pd.isna(raw_answer):
        return ""

    text = str(raw_answer).strip()

    # Keep first line only.
    text = text.split("\n")[0].strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .;:")

    if not text:
        return ""

    # Remove obvious answer prefixes.
    text = re.sub(
        r"^(answer:|the answer is|it is|it's|this is|that is)\s*",
        "",
        text,
        flags=re.I
    ).strip(" .;:")

    # Empty/refusal-like answers are unusable.
    if normalize_text(text) in {
        "unknown",
        "i dont know",
        "i do not know",
        "not sure",
        "none",
        "not available",
        "cannot determine",
        "cant determine",
        "no answer"
    }:
        return ""

    # If the model echoes the subject, keep it.
    # Subject echo can be a model error.
    if subject_label is not None:
        if normalize_text(text) == normalize_text(subject_label):
            return text

    # If the model answered in a sentence, extract entity part.
    extraction_patterns = [
        r"(?:was|is|were|are)\s+written\s+by\s+(.+)$",
        r"(?:was|is|were|are)\s+authored\s+by\s+(.+)$",
        r"(?:was|is|were|are)\s+composed\s+by\s+(.+)$",
        r"(?:was|is|were|are)\s+directed\s+by\s+(.+)$",
        r"(?:was|is|were|are)\s+produced\s+by\s+(.+)$",
        r"(?:was|is|were|are)\s+created\s+by\s+(.+)$",
        r"(?:was|is|were|are)\s+published\s+by\s+(.+)$",
        r"(?:was|is|were|are)\s+developed\s+by\s+(.+)$",
        r"(?:the author|the writer|the screenwriter|the producer|the director|the composer|the capital)\s+(?:is|was)\s+(.+)$",
        r"(?:author|writer|screenwriter|producer|director|composer|capital)\s+(?:is|was)\s+(.+)$",
        r"(?:written by|authored by|composed by|directed by|produced by|created by|published by|developed by)\s+(.+)$",
    ]

    for pattern in extraction_patterns:
        match = re.search(pattern, text, flags=re.I)

        if match:
            extracted = match.group(1).strip(" .;:")

            if extracted:
                text = extracted

            break

    # Remove explanation after answer.
    text = re.split(
        r"\s+(?:because|who|which|that|where|when|, and|\. )\s+",
        text,
        maxsplit=1,
        flags=re.I
    )[0]

    text = text.strip(" .;:")

    return text


def is_usable_model_answer(answer):
    """
    Usable means the answer can be evaluated as an entity-like answer.
    It does not mean correct.
    """
    if pd.isna(answer):
        return False

    answer = str(answer).strip()

    if not answer:
        return False

    norm = normalize_text(answer)

    if not norm:
        return False

    # Very long output probably means the model did not answer as an entity.
    if len(answer.split()) > 12:
        return False

    bad_fragments = {
        "the screenwriter",
        "the author",
        "the producer",
        "the director",
        "the composer",
        "written by",
        "authored by",
        "produced by",
        "directed by",
    }

    if norm in bad_fragments:
        return False

    return True


stage2_df["model_extracted_answer"] = stage2_df.apply(
    lambda row: extract_entity_answer(
        row["model_raw_answer"],
        subject_label=row["subject_label"]
    ),
    axis=1
)

stage2_df["is_model_answer_usable"] = stage2_df["model_extracted_answer"].apply(
    is_usable_model_answer
)

stage2_df["model_normalized_answer"] = stage2_df["model_extracted_answer"].apply(
    normalize_text
)


# ============================================================
# 9. Build reference answer set using reference_answer + PopQA aliases
# ============================================================

def build_reference_answer_entries(row):
    """
    Reference answer entries used for matching.

    Sources:
    - REFERENCE_ANSWER: main answer from PopQA object label
    - POPQA_POSSIBLE_ANSWER: PopQA possible_answers

    No manual aliases.
    No Wikidata aliases.
    """
    entries = []

    reference_answer = str(row.get("reference_answer", "")).strip()

    if reference_answer and reference_answer.lower() != "nan":
        entries.append({
            "answer": reference_answer,
            "source": ANSWER_MATCH_SOURCE["REFERENCE"]
        })

    for alias in parse_popqa_possible_answers(row.get("popqa_possible_answers", "")):
        alias = str(alias).strip()

        if alias and alias.lower() != "nan":
            entries.append({
                "answer": alias,
                "source": ANSWER_MATCH_SOURCE["POPQA"]
            })

    # Deduplicate.
    seen = set()
    clean_entries = []

    for entry in entries:
        answer = entry["answer"]
        source = entry["source"]

        forms = answer_forms(answer)

        if not forms:
            continue

        key = (normalize_text(answer), compact_text(answer), source)

        if key not in seen:
            seen.add(key)
            clean_entries.append(entry)

    return clean_entries


def build_reference_form_map(reference_entries):
    """
    Maps normalized answer forms to source.

    Priority:
    1. REFERENCE_ANSWER
    2. POPQA_POSSIBLE_ANSWER
    """
    source_priority = {
        ANSWER_MATCH_SOURCE["REFERENCE"]: 1,
        ANSWER_MATCH_SOURCE["POPQA"]: 2
    }

    form_map = {}

    for entry in reference_entries:
        answer = entry["answer"]
        source = entry["source"]

        for form in answer_forms(answer):
            if not form:
                continue

            if form not in form_map:
                form_map[form] = {
                    "answer": answer,
                    "source": source
                }
            else:
                old_source = form_map[form]["source"]

                if source_priority[source] < source_priority[old_source]:
                    form_map[form] = {
                        "answer": answer,
                        "source": source
                    }

    return form_map


reference_entries_list = []
reference_form_map_list = []

for _, row in stage2_df.iterrows():
    entries = build_reference_answer_entries(row)
    form_map = build_reference_form_map(entries)

    reference_entries_list.append(entries)
    reference_form_map_list.append(form_map)

stage2_df["reference_answer_entries"] = [
    json.dumps(x, ensure_ascii=False)
    for x in reference_entries_list
]

stage2_df["num_reference_answer_forms"] = [
    len(x)
    for x in reference_entries_list
]


# ============================================================
# 10. Match model answer to reference answer set
# ============================================================

def match_model_answer(row, reference_form_map):
    if not row["is_model_answer_usable"]:
        return {
            "model_answer_matches_reference": False,
            "matched_reference_answer": None,
            "answer_match_type": ANSWER_MATCH_TYPE["INVALID"],
            "answer_match_source": ANSWER_MATCH_SOURCE["INVALID"]
        }

    model_answer = row["model_extracted_answer"]
    model_forms = answer_forms(model_answer)

    for form in model_forms:
        if form in reference_form_map:
            matched = reference_form_map[form]
            source = matched["source"]

            if source == ANSWER_MATCH_SOURCE["REFERENCE"]:
                match_type = ANSWER_MATCH_TYPE["EXACT_REFERENCE"]
            elif source == ANSWER_MATCH_SOURCE["POPQA"]:
                match_type = ANSWER_MATCH_TYPE["POPQA_ALIAS"]
            else:
                match_type = source

            return {
                "model_answer_matches_reference": True,
                "matched_reference_answer": matched["answer"],
                "answer_match_type": match_type,
                "answer_match_source": source
            }

    return {
        "model_answer_matches_reference": False,
        "matched_reference_answer": None,
        "answer_match_type": ANSWER_MATCH_TYPE["NO_MATCH"],
        "answer_match_source": ANSWER_MATCH_SOURCE["NO_MATCH"]
    }


match_results = []

for idx, row in stage2_df.iterrows():
    result = match_model_answer(row, reference_form_map_list[idx])
    match_results.append(result)

match_df = pd.DataFrame(match_results)

for col in match_df.columns:
    stage2_df[col] = match_df[col]


# ============================================================
# 11. Assign final human-friendly answer case
# ============================================================

def assign_answer_case(row):
    """
    Final Stage 2 evaluation label.

    These labels are for evaluation only.
    They are not given to the hallucination detector.
    """
    if not row["is_model_answer_usable"]:
        return ANSWER_CASE["UNUSABLE"]

    if row["model_answer_matches_reference"]:
        if row["kg_fact_status"] == KG_FACT_STATUS["PRESENT"]:
            return ANSWER_CASE["CORRECT_PRESENT"]

        if row["kg_fact_status"] == KG_FACT_STATUS["MISSING"]:
            return ANSWER_CASE["CORRECT_MISSING"]

    return ANSWER_CASE["WRONG"]


stage2_df["answer_case"] = stage2_df.apply(assign_answer_case, axis=1)

stage2_df["is_exact_reference_match"] = (
    stage2_df["answer_match_source"] == ANSWER_MATCH_SOURCE["REFERENCE"]
)

stage2_df["is_popqa_alias_match"] = (
    stage2_df["answer_match_source"] == ANSWER_MATCH_SOURCE["POPQA"]
)


# ============================================================
# 12. Diagnostics
# ============================================================

print("\nFinal Stage 2 answer case distribution:")
print(stage2_df["answer_case"].value_counts())

print("\nReference match rate:")
print(round(float(stage2_df["model_answer_matches_reference"].mean()), 4))

print("\nAnswer match type distribution:")
print(stage2_df["answer_match_type"].value_counts())

print("\nAnswer match source distribution:")
print(stage2_df["answer_match_source"].value_counts())

print("\nPopQA alias match counts:")
print(stage2_df["is_popqa_alias_match"].value_counts())

relation_summary = (
    stage2_df
    .groupby("relation_normalized")
    .agg(
        num_rows=("question", "count"),
        correct_answers=("model_answer_matches_reference", lambda x: int((x == True).sum())),
        correct_and_fact_present_in_kg=("answer_case", lambda x: int((x == ANSWER_CASE["CORRECT_PRESENT"]).sum())),
        correct_but_fact_missing_from_kg=("answer_case", lambda x: int((x == ANSWER_CASE["CORRECT_MISSING"]).sum())),
        wrong_model_answers=("answer_case", lambda x: int((x == ANSWER_CASE["WRONG"]).sum())),
        unusable_model_answers=("answer_case", lambda x: int((x == ANSWER_CASE["UNUSABLE"]).sum())),
        exact_reference_matches=("is_exact_reference_match", lambda x: int((x == True).sum())),
        popqa_alias_matches=("is_popqa_alias_match", lambda x: int((x == True).sum())),
    )
    .reset_index()
)

print("\nRelation-level Stage 2 summary:")
display(relation_summary.sort_values("correct_answers", ascending=False))

print("\nStage 2 preview:")
display(stage2_df[[
    "question",
    "subject_label",
    "relation_normalized",
    "reference_answer",
    "popqa_possible_answers",
    "model_raw_answer",
    "model_extracted_answer",
    "is_model_answer_usable",
    "model_answer_matches_reference",
    "matched_reference_answer",
    "answer_match_type",
    "answer_match_source",
    "kg_fact_status",
    "answer_case",
    "subject_popularity_bin"
]].head(50))


# ============================================================
# 13. Save PopQA alias audit
# ============================================================

alias_audit_df = stage2_df[
    stage2_df["is_popqa_alias_match"] == True
][[
    "question",
    "subject_label",
    "relation_normalized",
    "reference_answer",
    "popqa_possible_answers",
    "model_raw_answer",
    "model_extracted_answer",
    "matched_reference_answer",
    "answer_match_type",
    "answer_match_source",
    "kg_fact_status",
    "answer_case",
    "subject_popularity_bin"
]].copy()

alias_audit_df.to_csv(ALIAS_AUDIT_PATH, index=False)

print("\nSaved PopQA alias audit:", ALIAS_AUDIT_PATH)
print("PopQA alias audit rows:", len(alias_audit_df))

display(alias_audit_df.head(50))


# ============================================================
# 14. Save final Stage 2 output
# ============================================================

# Remove old-style labels if they came from any reused file.
old_label_columns_to_remove = [
    "gold_kg_label",
    "genai_case",
    "genai_matches_gold",
    "match_type",
    "match_source",
    "best_gold_match",
    "llm_raw_answer",
    "llm_clean_answer",
    "llm_extracted_answer"
]

stage2_df = stage2_df.drop(columns=old_label_columns_to_remove, errors="ignore")

stage2_df.to_csv(OUTPUT_PATH, index=False)

metadata = {
    "stage": "Stage 2 Final Model Answer Generation and PopQA Alias Evaluation",
    "input_file": STAGE1_PATH,
    "output_file": OUTPUT_PATH,
    "alias_audit_file": ALIAS_AUDIT_PATH,
    "model_name": GENAI_MODEL_NAME,
    "device": device,
    "num_rows_requested": int(NUM_ROWS_FOR_STAGE2),
    "num_rows_in_stage2": int(len(stage2_df)),
    "num_generated_now": int(num_generated_now),
    "force_regenerate_model_answers": bool(FORCE_REGENERATE_MODEL_ANSWERS),
    "batch_size_used": None if batch_size_used is None else int(batch_size_used),
    "elapsed_seconds_for_generation": float(elapsed_seconds),
    "alias_policy": {
        "manual_alias_map_used": False,
        "wikidata_aliases_used": False,
        "popqa_possible_answers_used": True,
        "main_reference_answer_used": True
    },
    "label_policy": {
        "old_technical_labels_removed": True,
        "human_friendly_labels_used": True
    },
    "kg_fact_status_distribution": stage2_df["kg_fact_status"].value_counts().to_dict(),
    "answer_case_distribution": stage2_df["answer_case"].value_counts().to_dict(),
    "answer_match_type_distribution": stage2_df["answer_match_type"].value_counts().to_dict(),
    "answer_match_source_distribution": stage2_df["answer_match_source"].value_counts().to_dict(),
    "reference_match_rate": float(stage2_df["model_answer_matches_reference"].mean()),
    "relation_distribution": stage2_df["relation_normalized"].value_counts().to_dict(),
    "subject_popularity_distribution": stage2_df["subject_popularity_bin"].value_counts().astype(int).to_dict(),
    "relation_summary": relation_summary.to_dict(orient="records"),
    "label_definitions": {
        "kg_fact_status": {
            KG_FACT_STATUS["PRESENT"]: "The true reference fact exists in the incomplete KG.",
            KG_FACT_STATUS["MISSING"]: "The true reference fact was removed from the incomplete KG."
        },
        "answer_case": {
            ANSWER_CASE["CORRECT_PRESENT"]: "The model answered correctly and the fact is present in the incomplete KG.",
            ANSWER_CASE["CORRECT_MISSING"]: "The model answered correctly but the fact is missing from the incomplete KG.",
            ANSWER_CASE["WRONG"]: "The model answer does not match the reference answer or PopQA aliases.",
            ANSWER_CASE["UNUSABLE"]: "The model output is not usable as a short entity answer."
        },
        "answer_match_type": {
            ANSWER_MATCH_TYPE["EXACT_REFERENCE"]: "The model answer exactly matched the main reference answer.",
            ANSWER_MATCH_TYPE["POPQA_ALIAS"]: "The model answer matched one of PopQA's possible_answers.",
            ANSWER_MATCH_TYPE["NO_MATCH"]: "The model answer did not match the reference answer or PopQA aliases.",
            ANSWER_MATCH_TYPE["INVALID"]: "The model output could not be evaluated."
        }
    },
    "notes": {
        "stage_objective": (
            "Stage 2 obtains real model answers and labels them against PopQA references. "
            "The labels are used only for evaluation, not as input to later detectors."
        ),
        "why_popqa_aliases": (
            "PopQA possible_answers are dataset-provided and reproducible, making them safer "
            "than manual or live external aliases for primary evaluation."
        )
    }
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved Stage 2 output:", OUTPUT_PATH)
print("Saved Stage 2 metadata:", METADATA_PATH)

print("\nMetadata:")
print(json.dumps(metadata, indent=2))
# Stage 3
# ============================================================
# STAGE 3 FINAL:
# KG-only hallucination baseline on Stage 2 model answers
#
# Input:
# - data_final/stage2_model_answers.csv
# - data_final/kg_incomplete_clean.csv
#
# Output:
# - data_final/stage3_kg_only_predictions.csv
# - data_final/stage3_kg_only_metrics.json
# ============================================================

!pip install -q pandas numpy scikit-learn

import os
import json
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# 0. Config
# ============================================================

DATA_DIR = "data_final"

STAGE2_PATH = f"{DATA_DIR}/stage2_model_answers.csv"
KG_INCOMPLETE_PATH = f"{DATA_DIR}/kg_incomplete_clean.csv"

STAGE3_OUTPUT_PATH = f"{DATA_DIR}/stage3_kg_only_predictions.csv"
STAGE3_METRICS_PATH = f"{DATA_DIR}/stage3_kg_only_metrics.json"

assert os.path.exists(STAGE2_PATH), f"Stage 2 output not found: {STAGE2_PATH}"
assert os.path.exists(KG_INCOMPLETE_PATH), f"Incomplete KG file not found: {KG_INCOMPLETE_PATH}"


# ============================================================
# 1. Load inputs
# ============================================================

stage2_df = pd.read_csv(STAGE2_PATH)
kg_incomplete = pd.read_csv(KG_INCOMPLETE_PATH)

print("Stage 2 rows:", len(stage2_df))
print("Incomplete KG triples:", len(kg_incomplete))

required_stage2_cols = [
    "triple_id",
    "question",
    "subject_label",
    "relation_normalized",
    "reference_answer",
    "model_raw_answer",
    "model_extracted_answer",
    "is_model_answer_usable",
    "model_answer_matches_reference",
    "kg_fact_status",
    "answer_case",
    "subject_popularity_bin"
]

missing_cols = [c for c in required_stage2_cols if c not in stage2_df.columns]
if missing_cols:
    raise ValueError(f"Missing expected Stage 2 columns: {missing_cols}")

if "triple_id" not in kg_incomplete.columns:
    raise ValueError("kg_incomplete_clean.csv must contain a triple_id column.")

print("\nStage 2 answer case distribution:")
print(stage2_df["answer_case"].value_counts())

print("\nStage 2 KG fact status distribution:")
print(stage2_df["kg_fact_status"].value_counts())

kg_triple_set = set(kg_incomplete["triple_id"].astype(str))


# ============================================================
# 2. Helper functions
# ============================================================

def to_bool(value):
    """
    Converts booleans safely when CSV reload turns True/False into strings.
    """
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    value = str(value).strip().lower()

    if value in {"true", "1", "yes", "y"}:
        return True

    return False


def kg_status_from_triple(row):
    """
    Recompute whether the true reference fact is present in the incomplete KG.
    This avoids relying only on the Stage 2 saved label.
    """
    triple_id = str(row["triple_id"])

    if triple_id in kg_triple_set:
        return "FACT_PRESENT_IN_KG"

    return "FACT_MISSING_FROM_KG"


def assign_stage3_answer_case(row):
    """
    Human-readable analysis case for Stage 3.
    This mirrors Stage 2 logic, but recomputes it from current columns.
    """

    usable = to_bool(row["is_model_answer_usable"])
    matches_reference = to_bool(row["model_answer_matches_reference"])
    kg_status = row["kg_fact_status_checked"]

    if not usable:
        return "UNUSABLE_MODEL_ANSWER"

    if matches_reference and kg_status == "FACT_PRESENT_IN_KG":
        return "CORRECT_AND_FACT_PRESENT_IN_KG"

    if matches_reference and kg_status == "FACT_MISSING_FROM_KG":
        return "CORRECT_BUT_FACT_MISSING_FROM_KG"

    return "WRONG_MODEL_ANSWER"


def kg_only_predict(row):
    """
    KG-only baseline.

    Core idea:
    - If model output is unusable, mark it unusable.
    - If model answer is correct AND the fact exists in the incomplete KG,
      KG-only marks it as supported.
    - Otherwise, KG-only flags it as hallucination.

    This intentionally exposes the weakness of KG-only detection:
    a correct answer is still flagged if the KG is incomplete.
    """

    if row["stage3_answer_case"] == "UNUSABLE_MODEL_ANSWER":
        return "UNUSABLE_MODEL_ANSWER"

    if (
        to_bool(row["model_answer_matches_reference"])
        and row["kg_fact_status_checked"] == "FACT_PRESENT_IN_KG"
    ):
        return "SUPPORTED_BY_KG_ONLY"

    return "FLAGGED_AS_HALLUCINATION_BY_KG_ONLY"


# ============================================================
# 3. Recheck KG status and run KG-only prediction
# ============================================================

stage3_df = stage2_df.copy()

stage3_df["kg_fact_status_checked"] = stage3_df.apply(
    kg_status_from_triple,
    axis=1
)

status_mismatches = stage3_df[
    stage3_df["kg_fact_status"].astype(str) != stage3_df["kg_fact_status_checked"].astype(str)
].copy()

print("\nKG status mismatches between Stage 2 and recomputed KG:")
print(len(status_mismatches))

if len(status_mismatches) > 0:
    print("Warning: Some Stage 2 kg_fact_status values do not match kg_incomplete_clean.csv.")
    display(status_mismatches[[
        "question",
        "triple_id",
        "kg_fact_status",
        "kg_fact_status_checked"
    ]].head(20))

stage3_df["stage3_answer_case"] = stage3_df.apply(
    assign_stage3_answer_case,
    axis=1
)

stage3_df["kg_only_prediction"] = stage3_df.apply(
    kg_only_predict,
    axis=1
)

print("\nStage 3 answer case distribution:")
print(stage3_df["stage3_answer_case"].value_counts())

print("\nKG-only prediction distribution:")
print(stage3_df["kg_only_prediction"].value_counts())


# ============================================================
# 4. Binary evaluation
# ============================================================

eval_df = stage3_df[
    stage3_df["stage3_answer_case"] != "UNUSABLE_MODEL_ANSWER"
].copy()

eval_df["reference_binary_label"] = np.where(
    eval_df["model_answer_matches_reference"].apply(to_bool),
    "NOT_HALLUCINATION",
    "HALLUCINATION"
)

eval_df["kg_only_binary_prediction"] = np.where(
    eval_df["kg_only_prediction"] == "SUPPORTED_BY_KG_ONLY",
    "NOT_HALLUCINATION",
    "HALLUCINATION"
)

labels = ["NOT_HALLUCINATION", "HALLUCINATION"]

accuracy = accuracy_score(
    eval_df["reference_binary_label"],
    eval_df["kg_only_binary_prediction"]
)

true_not_hallucination = eval_df["reference_binary_label"] == "NOT_HALLUCINATION"
predicted_hallucination = eval_df["kg_only_binary_prediction"] == "HALLUCINATION"

false_hallucinations = eval_df[
    true_not_hallucination & predicted_hallucination
].copy()

false_hallucination_rate = (
    len(false_hallucinations) / int(true_not_hallucination.sum())
    if int(true_not_hallucination.sum()) > 0
    else 0.0
)

correct_but_missing = eval_df[
    eval_df["stage3_answer_case"] == "CORRECT_BUT_FACT_MISSING_FROM_KG"
].copy()

print("\n================ KG-ONLY BASELINE METRICS ================")
print("Evaluated rows:", len(eval_df))
print("Accuracy:", round(accuracy, 4))

print("\nCorrect model answers:", int(true_not_hallucination.sum()))
print("False hallucinations by KG-only:", len(false_hallucinations))
print("False hallucination rate:", round(false_hallucination_rate, 4))

print("\nCorrect-but-fact-missing cases:", len(correct_but_missing))
print("KG-only predictions on correct-but-fact-missing:")
print(correct_but_missing["kg_only_prediction"].value_counts())

print("\nClassification report:")
print(classification_report(
    eval_df["reference_binary_label"],
    eval_df["kg_only_binary_prediction"],
    labels=labels,
    zero_division=0
))

print("\nConfusion matrix:")
cm = confusion_matrix(
    eval_df["reference_binary_label"],
    eval_df["kg_only_binary_prediction"],
    labels=labels
)

display(pd.DataFrame(
    cm,
    index=[f"true_{x}" for x in labels],
    columns=[f"pred_{x}" for x in labels]
))


# ============================================================
# 5. Breakdown by relation
# ============================================================

relation_breakdown_rows = []

for relation, group in eval_df.groupby("relation_normalized"):
    true_correct = group["reference_binary_label"] == "NOT_HALLUCINATION"
    false_hall = group[
        true_correct & (group["kg_only_binary_prediction"] == "HALLUCINATION")
    ]

    relation_breakdown_rows.append({
        "relation": relation,
        "num_rows": int(len(group)),
        "correct_model_answers": int(true_correct.sum()),
        "kg_only_false_hallucinations": int(len(false_hall)),
        "kg_only_false_hallucination_rate": float(
            len(false_hall) / true_correct.sum()
            if true_correct.sum() > 0
            else 0.0
        ),
        "correct_but_fact_missing_cases": int(
            (group["stage3_answer_case"] == "CORRECT_BUT_FACT_MISSING_FROM_KG").sum()
        )
    })

relation_breakdown = pd.DataFrame(relation_breakdown_rows).sort_values(
    "kg_only_false_hallucination_rate",
    ascending=False
)

print("\nFalse hallucination breakdown by relation:")
display(relation_breakdown)


# ============================================================
# 6. Breakdown by subject popularity
# ============================================================

popularity_breakdown_rows = []

for pop_bin, group in eval_df.groupby("subject_popularity_bin"):
    true_correct = group["reference_binary_label"] == "NOT_HALLUCINATION"
    false_hall = group[
        true_correct & (group["kg_only_binary_prediction"] == "HALLUCINATION")
    ]

    popularity_breakdown_rows.append({
        "subject_popularity_bin": pop_bin,
        "num_rows": int(len(group)),
        "correct_model_answers": int(true_correct.sum()),
        "kg_only_false_hallucinations": int(len(false_hall)),
        "kg_only_false_hallucination_rate": float(
            len(false_hall) / true_correct.sum()
            if true_correct.sum() > 0
            else 0.0
        ),
        "correct_but_fact_missing_cases": int(
            (group["stage3_answer_case"] == "CORRECT_BUT_FACT_MISSING_FROM_KG").sum()
        )
    })

popularity_breakdown = pd.DataFrame(popularity_breakdown_rows).sort_values(
    "kg_only_false_hallucination_rate",
    ascending=False
)

print("\nFalse hallucination breakdown by subject popularity:")
display(popularity_breakdown)


# ============================================================
# 7. Important failure examples
# ============================================================

print("\nExamples where KG-only falsely marks correct model answers as hallucinations:")

example_cols = [
    "question",
    "subject_label",
    "relation_label",
    "relation_normalized",
    "reference_answer",
    "model_raw_answer",
    "model_extracted_answer",
    "kg_fact_status_checked",
    "stage3_answer_case",
    "kg_only_prediction",
    "subject_popularity_bin"
]

example_cols = [c for c in example_cols if c in false_hallucinations.columns]

display(false_hallucinations[example_cols].head(30))


# ============================================================
# 8. Save outputs
# ============================================================

stage3_df.to_csv(STAGE3_OUTPUT_PATH, index=False)

metrics = {
    "stage": "Stage 3 Final KG-only Baseline",
    "input_stage2_file": STAGE2_PATH,
    "input_incomplete_kg_file": KG_INCOMPLETE_PATH,
    "output_file": STAGE3_OUTPUT_PATH,

    "num_total_rows": int(len(stage3_df)),
    "num_evaluated_rows": int(len(eval_df)),
    "kg_size_incomplete": int(len(kg_incomplete)),

    "accuracy": float(accuracy),

    "correct_model_answers": int(true_not_hallucination.sum()),
    "false_hallucination_count": int(len(false_hallucinations)),
    "false_hallucination_rate": float(false_hallucination_rate),
    "correct_but_fact_missing_cases": int(len(correct_but_missing)),

    "stage2_answer_case_distribution": stage2_df["answer_case"].value_counts().to_dict(),
    "stage3_answer_case_distribution": stage3_df["stage3_answer_case"].value_counts().to_dict(),
    "kg_fact_status_distribution": stage3_df["kg_fact_status_checked"].value_counts().to_dict(),
    "kg_only_prediction_distribution": stage3_df["kg_only_prediction"].value_counts().to_dict(),

    "kg_status_mismatch_count": int(len(status_mismatches)),

    "relation_breakdown": relation_breakdown.to_dict(orient="records"),
    "subject_popularity_breakdown": popularity_breakdown.to_dict(orient="records"),

    "classification_report": classification_report(
        eval_df["reference_binary_label"],
        eval_df["kg_only_binary_prediction"],
        labels=labels,
        zero_division=0,
        output_dict=True
    ),

    "confusion_matrix_labels": labels,
    "confusion_matrix": cm.tolist(),

    "label_definitions": {
        "stage3_answer_case": {
            "CORRECT_AND_FACT_PRESENT_IN_KG": "The model answer is correct and the reference fact exists in the incomplete KG.",
            "CORRECT_BUT_FACT_MISSING_FROM_KG": "The model answer is correct but the reference fact was removed from the incomplete KG.",
            "WRONG_MODEL_ANSWER": "The model answer does not match the reference answer or PopQA aliases.",
            "UNUSABLE_MODEL_ANSWER": "The model output is not usable as a short entity answer."
        },
        "kg_only_prediction": {
            "SUPPORTED_BY_KG_ONLY": "The KG-only baseline marks the answer as supported because the correct reference triple exists in the incomplete KG.",
            "FLAGGED_AS_HALLUCINATION_BY_KG_ONLY": "The KG-only baseline marks the answer as hallucination because the fact is missing from the incomplete KG or the model answer is wrong.",
            "UNUSABLE_MODEL_ANSWER": "The model output was not evaluated because it is unusable."
        },
        "reference_binary_label": {
            "NOT_HALLUCINATION": "The model answer matches the PopQA reference answer or PopQA aliases.",
            "HALLUCINATION": "The model answer does not match the PopQA reference answer or PopQA aliases."
        }
    },

    "notes": {
        "core_baseline_logic": "KG-only detection treats an answer as supported only when the model answer is correct and the reference triple is present in the incomplete KG.",
        "known_limitation": "This baseline intentionally exposes KG incompleteness: correct answers whose facts were removed from the KG are falsely flagged as hallucinations.",
        "no_old_stage2_labels_used": True,
        "old_labels_removed": [
            "genai_case",
            "gold_kg_label",
            "KG_SUPPORTED",
            "KG_ONLY_HALLUCINATED",
            "INVALID_GENAI_ANSWER"
        ]
    }
}

with open(STAGE3_METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved Stage 3 predictions:", STAGE3_OUTPUT_PATH)
print("Saved Stage 3 metrics:", STAGE3_METRICS_PATH)

print("Key metrics:")
print("Accuracy:", round(accuracy, 4))
print("False hallucination count:", len(false_hallucinations))
print("False hallucination rate:", round(false_hallucination_rate, 4))
print("Correct-but-fact-missing cases:", len(correct_but_missing))
# Stage 4
# ============================================================
# STAGE 4 FINAL CLEAN:
# Wikipedia evidence retrieval for KG-only flagged model answers
#
# Input:
# - data_final/stage3_kg_only_predictions.csv
#
# Output:
# - data_final/stage4_claims_with_evidence.csv
# - data_final/stage4_evidence_corpus.csv
# - data_final/stage4_wikipedia_cache.json
# - data_final/stage4_retrieval_metadata.json
#
# Main fixes:
# - row_index now points to the ORIGINAL Stage 3 row index.
# - Uses current human-friendly Stage 2/3 labels.
# - Keeps Wikipedia-only evidence source.
# - Adds answer/relation grounding filters.
# - Keeps relation-only evidence because it is useful for contradiction detection.
# - Removes unnecessary old labels and old output naming.
# ============================================================

!pip install -q pandas numpy tqdm wikipedia-api rank_bm25 sentence-transformers

import os
import re
import ast
import json
import time
import gc
import unicodedata
import numpy as np
import pandas as pd
import torch

from tqdm.auto import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import wikipediaapi


# ============================================================
# 0. Free GPU memory if previous models are still loaded
# ============================================================

for obj_name in ["model", "tokenizer", "embedding_model", "reranker_model"]:
    try:
        del globals()[obj_name]
    except Exception:
        pass

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# 1. Config
# ============================================================

DATA_DIR = "data_final"

STAGE3_PATH = f"{DATA_DIR}/stage3_kg_only_predictions.csv"

CACHE_PATH = f"{DATA_DIR}/stage4_wikipedia_cache.json"
EVIDENCE_CORPUS_PATH = f"{DATA_DIR}/stage4_evidence_corpus.csv"
STAGE4_OUTPUT_PATH = f"{DATA_DIR}/stage4_claims_with_evidence.csv"
STAGE4_METADATA_PATH = f"{DATA_DIR}/stage4_retrieval_metadata.json"

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

BM25_CANDIDATES = 30
DENSE_CANDIDATES = 30
RERANK_CANDIDATES = 40
TOP_EVIDENCE_PER_CLAIM = 5

KG_ONLY_FLAGGED_LABEL = "FLAGGED_AS_HALLUCINATION_BY_KG_ONLY"
KG_ONLY_SUPPORTED_LABEL = "SUPPORTED_BY_KG_ONLY"
UNUSABLE_ANSWER_CASE = "UNUSABLE_MODEL_ANSWER"

assert os.path.exists(STAGE3_PATH), f"Stage 3 output not found: {STAGE3_PATH}"

print("Using Stage 3 file:", STAGE3_PATH)


# ============================================================
# 2. Load Stage 3 and validate columns
# ============================================================

stage3_df = pd.read_csv(STAGE3_PATH)

# Preserve original Stage 3 row index.
# This is the key fix.
if "row_index" in stage3_df.columns:
    stage3_df = stage3_df.rename(columns={"row_index": "old_row_index"})

stage3_df.insert(0, "row_index", range(len(stage3_df)))

required_cols = [
    "row_index",
    "id",
    "question",
    "subject_label",
    "relation_label",
    "relation_normalized",
    "reference_answer",
    "popqa_possible_answers",
    "s_wiki_title",
    "subject_popularity_bin",
    "model_raw_answer",
    "model_extracted_answer",
    "model_answer_matches_reference",
    "answer_case",
    "stage3_answer_case",
    "kg_fact_status_checked",
    "kg_only_prediction"
]

missing_cols = [c for c in required_cols if c not in stage3_df.columns]
if missing_cols:
    raise ValueError(f"Missing required Stage 3 columns: {missing_cols}")

print("Stage 3 rows:", len(stage3_df))

print("\nKG-only prediction distribution:")
print(stage3_df["kg_only_prediction"].value_counts())

print("\nStage 3 answer case distribution:")
print(stage3_df["stage3_answer_case"].value_counts())

print("\nKG fact status distribution:")
print(stage3_df["kg_fact_status_checked"].value_counts())


# ============================================================
# 3. Select rows needing evidence verification
# ============================================================

verify_df = stage3_df[
    (stage3_df["kg_only_prediction"] == KG_ONLY_FLAGGED_LABEL) &
    (stage3_df["stage3_answer_case"] != UNUSABLE_ANSWER_CASE)
].copy()

# Keep original Stage 3 row_index.
# Add separate verification_row_index only for audit/debugging.
verify_df = verify_df.reset_index(drop=True)
verify_df["verification_row_index"] = verify_df.index

print("\nRows needing evidence verification:", len(verify_df))
print(verify_df["stage3_answer_case"].value_counts())


# ============================================================
# 4. Text normalization helpers
# ============================================================

def strip_accents(text):
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).strip()
    text = strip_accents(text)
    text = text.lower()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[\“\”\‘\’\"']", "", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def compact_text(text):
    return re.sub(r"[^a-z0-9]", "", normalize_text(text))


def parse_list_like(value):
    if isinstance(value, list):
        return [str(x) for x in value]

    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        if isinstance(parsed, tuple):
            return [str(x) for x in parsed]
        return [str(parsed)]
    except Exception:
        text = text.strip("[]")
        parts = re.split(r",|\|", text)
        return [p.strip().strip("'").strip('"') for p in parts if p.strip()]


def clean_answer_for_claim(answer):
    if pd.isna(answer):
        return ""

    answer = str(answer).strip()
    answer = answer.split("\n")[0].strip()
    answer = answer.replace("_", " ")
    answer = re.sub(r"\s+", " ", answer)
    answer = answer.strip(" .;:")

    return answer


def make_search_terms(values):
    terms = set()

    for value in values:
        if pd.isna(value):
            continue

        raw = str(value).strip()
        if not raw:
            continue

        norm = normalize_text(raw)
        comp = compact_text(raw)

        if norm:
            terms.add(norm)

        if comp and len(comp) >= 3:
            terms.add(comp)

    return sorted(terms, key=len, reverse=True)


def text_contains_term(text, term):
    if not term:
        return False

    norm_text = normalize_text(text)
    comp_text = compact_text(text)

    norm_term = normalize_text(term)
    comp_term = compact_text(term)

    if not norm_term and not comp_term:
        return False

    # For very short terms, require word-boundary match.
    if len(comp_term) < 4:
        pattern = r"\b" + re.escape(norm_term) + r"\b"
        return re.search(pattern, norm_text) is not None

    if norm_term and norm_term in norm_text:
        return True

    if comp_term and comp_term in comp_text:
        return True

    return False


def text_contains_any_term(text, terms):
    for term in terms:
        if text_contains_term(text, term):
            return True, term
    return False, ""


# ============================================================
# 5. Claim creation
# ============================================================

def verbalize_claim(subject, relation, answer):
    subject = str(subject).strip()
    relation = str(relation).strip().lower()
    answer = clean_answer_for_claim(answer)

    templates = {
        "author": "{subject} was written by {answer}.",
        "composer": "{subject} was composed by {answer}.",
        "director": "{subject} was directed by {answer}.",
        "producer": "{subject} was produced by {answer}.",
        "screenwriter": "{subject} was written by {answer}.",
        "capital": "The capital of {subject} is {answer}.",
        "place of birth": "{subject} was born in {answer}.",
        "country": "{subject} is associated with the country {answer}.",
        "developer": "{subject} was developed by {answer}.",
        "publisher": "{subject} was published by {answer}."
    }

    if relation in templates:
        return templates[relation].format(subject=subject, answer=answer)

    return f"The {relation} of {subject} is {answer}."


verify_df["model_clean_answer"] = verify_df["model_extracted_answer"].apply(clean_answer_for_claim)

verify_df["claim"] = verify_df.apply(
    lambda row: verbalize_claim(
        row["subject_label"],
        row["relation_normalized"],
        row["model_clean_answer"]
    ),
    axis=1
)

# Remove rows with empty model answer.
verify_df = verify_df[
    verify_df["model_clean_answer"].astype(str).str.strip() != ""
].copy().reset_index(drop=True)

verify_df["verification_row_index"] = verify_df.index

print("\nRows after removing empty claims:", len(verify_df))


# ============================================================
# 6. Build grounding terms
# ============================================================

def build_answer_grounding_terms(row):
    """
    For wrong model answers:
    - Use only the model answer.

    For correct-but-KG-missing answers:
    - Use model answer plus PopQA reference/possible answers.
    - This helps cases like USA vs United States.

    No manual alias dictionary is used.
    """

    terms = []

    model_answer = clean_answer_for_claim(row["model_clean_answer"])
    if model_answer:
        terms.append(model_answer)

    model_matches_reference = bool(row["model_answer_matches_reference"])

    if model_matches_reference:
        terms.append(str(row.get("reference_answer", "")))

        matched = row.get("matched_reference_answer", "")
        if not pd.isna(matched) and str(matched).strip():
            terms.append(str(matched))

        for ans in parse_list_like(row.get("popqa_possible_answers", "")):
            terms.append(ans)

    return make_search_terms(terms)


RELATION_KEYWORDS = {
    "author": [
        "written by", "authored by", "author", "writer", "novel by", "book by"
    ],
    "composer": [
        "composed by", "composer", "music by", "written by", "song by"
    ],
    "director": [
        "directed by", "director"
    ],
    "producer": [
        "produced by", "producer"
    ],
    "screenwriter": [
        "written by", "screenplay by", "screenwriter", "writer", "script by"
    ],
    "capital": [
        "capital", "capital city", "administrative centre", "administrative center", "seat"
    ],
    "place of birth": [
        "born in", "born at", "birthplace", "place of birth"
    ],
    "country": [
        "country", "located in", "based in", "headquartered in",
        "village in", "town in", "city in", "commune in",
        "airport in", "river in", "municipality in", "province in"
    ],
    "developer": [
        "developed by", "developer", "created by"
    ],
    "publisher": [
        "published by", "publisher"
    ]
}


def build_relation_grounding_terms(relation):
    relation = str(relation).strip().lower()
    terms = RELATION_KEYWORDS.get(relation, [relation])
    return make_search_terms(terms)


verify_df["answer_grounding_terms"] = verify_df.apply(
    build_answer_grounding_terms,
    axis=1
)

verify_df["relation_grounding_terms"] = verify_df["relation_normalized"].apply(
    build_relation_grounding_terms
)

display(verify_df[[
    "row_index",
    "verification_row_index",
    "question",
    "subject_label",
    "relation_normalized",
    "reference_answer",
    "model_raw_answer",
    "model_clean_answer",
    "claim",
    "stage3_answer_case",
    "kg_fact_status_checked",
    "s_wiki_title",
    "answer_grounding_terms"
]].head(15))


# ============================================================
# 7. Fetch subject Wikipedia pages
# ============================================================

wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="KGIncompletenessHallucinationResearch/1.0"
)

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r") as f:
        wiki_cache = json.load(f)
else:
    wiki_cache = {}

print("\nCached pages before fetching:", len(wiki_cache))


def fetch_wikipedia_page_text(title, sleep_time=0.05):
    if pd.isna(title) or str(title).strip() == "":
        return ""

    title = str(title).strip()

    if title in wiki_cache:
        return wiki_cache[title]

    try:
        page = wiki.page(title)

        if page.exists():
            text = page.text or ""
        else:
            text = ""

        wiki_cache[title] = text
        time.sleep(sleep_time)

        return text

    except Exception as e:
        print(f"Error fetching {title}: {e}")
        wiki_cache[title] = ""
        return ""


unique_titles = sorted(
    verify_df["s_wiki_title"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)

print("Unique subject Wikipedia titles:", len(unique_titles))

for title in tqdm(unique_titles):
    _ = fetch_wikipedia_page_text(title)

with open(CACHE_PATH, "w") as f:
    json.dump(wiki_cache, f)

print("Cached pages after fetching:", len(wiki_cache))
print("Saved cache:", CACHE_PATH)


# ============================================================
# 8. Build evidence corpus
# ============================================================

def clean_wiki_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def chunk_text(text, max_words=120, min_words=20):
    text = clean_wiki_text(text)

    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []

    for para in paragraphs:
        words = para.split()

        if len(words) < min_words:
            continue

        if len(words) <= max_words:
            chunks.append(para)
        else:
            step = max_words // 2

            for start in range(0, len(words), step):
                sub_words = words[start:start + max_words]

                if len(sub_words) >= min_words:
                    chunks.append(" ".join(sub_words))

    return chunks


corpus_rows = []

needed_title_set = set(unique_titles)

for title in tqdm(unique_titles):
    text = wiki_cache.get(title, "")
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        corpus_rows.append({
            "doc_id": f"{title}::chunk_{i}",
            "wiki_title": title,
            "chunk_index": i,
            "text": chunk
        })

evidence_corpus = pd.DataFrame(corpus_rows)

if len(evidence_corpus) == 0:
    raise ValueError("No evidence chunks created. Check Wikipedia page fetching.")

evidence_corpus.to_csv(EVIDENCE_CORPUS_PATH, index=False)

print("\nEvidence chunks:", len(evidence_corpus))
print("Saved evidence corpus:", EVIDENCE_CORPUS_PATH)

display(evidence_corpus.head(5))


# ============================================================
# 9. Load retrieval models
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("\nDevice:", device)

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
reranker_model = CrossEncoder(RERANKER_MODEL_NAME, device=device)

print("Loaded embedding model:", EMBEDDING_MODEL_NAME)
print("Loaded reranker model:", RERANKER_MODEL_NAME)


# ============================================================
# 10. Build dense embeddings for corpus
# ============================================================

corpus_texts = evidence_corpus["text"].fillna("").astype(str).tolist()

corpus_embeddings = embedding_model.encode(
    corpus_texts,
    batch_size=32 if device == "cuda" else 8,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Corpus embeddings shape:", corpus_embeddings.shape)


# ============================================================
# 11. Retrieval helpers
# ============================================================

def bm25_tokenize(text):
    text = normalize_text(text)
    return text.split()


def retrieve_subject_page_candidates(row, bm25_k=30, dense_k=30):
    query = row["claim"]
    subject_title = str(row.get("s_wiki_title", "")).strip()

    subject_chunks = evidence_corpus[
        evidence_corpus["wiki_title"].astype(str).str.strip() == subject_title
    ].copy()

    if len(subject_chunks) == 0:
        return []

    local_indices = subject_chunks.index.to_numpy()
    local_texts = subject_chunks["text"].fillna("").astype(str).tolist()

    # BM25 retrieval
    local_tokens = [bm25_tokenize(text) for text in local_texts]
    local_bm25 = BM25Okapi(local_tokens)
    query_tokens = bm25_tokenize(query)
    bm25_scores = local_bm25.get_scores(query_tokens)

    bm25_order = np.argsort(bm25_scores)[::-1][:min(bm25_k, len(local_indices))]

    # Dense retrieval
    local_embeddings = corpus_embeddings[local_indices]

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    dense_scores = np.dot(local_embeddings, query_embedding)
    dense_order = np.argsort(dense_scores)[::-1][:min(dense_k, len(local_indices))]

    merged = {}

    for local_pos in bm25_order:
        global_idx = int(local_indices[local_pos])
        merged[global_idx] = {
            "global_idx": global_idx,
            "doc_id": evidence_corpus.loc[global_idx, "doc_id"],
            "wiki_title": evidence_corpus.loc[global_idx, "wiki_title"],
            "text": evidence_corpus.loc[global_idx, "text"],
            "bm25_score": float(bm25_scores[local_pos]),
            "dense_score": float(dense_scores[local_pos]),
        }

    for local_pos in dense_order:
        global_idx = int(local_indices[local_pos])

        if global_idx not in merged:
            merged[global_idx] = {
                "global_idx": global_idx,
                "doc_id": evidence_corpus.loc[global_idx, "doc_id"],
                "wiki_title": evidence_corpus.loc[global_idx, "wiki_title"],
                "text": evidence_corpus.loc[global_idx, "text"],
                "bm25_score": float(bm25_scores[local_pos]),
                "dense_score": float(dense_scores[local_pos]),
            }
        else:
            merged[global_idx]["dense_score"] = float(dense_scores[local_pos])

    return list(merged.values())


def rerank_candidates(query, candidates, max_candidates=40):
    if len(candidates) == 0:
        return []

    # Keep candidate count bounded.
    candidates = candidates[:max_candidates]

    pairs = [[query, c["text"]] for c in candidates]

    scores = reranker_model.predict(
        pairs,
        batch_size=16 if device == "cuda" else 4,
        show_progress_bar=False
    )

    for c, score in zip(candidates, scores):
        c["reranker_score"] = float(score)

    return sorted(
        candidates,
        key=lambda x: x["reranker_score"],
        reverse=True
    )


def classify_evidence_strength(evidence_text, answer_terms, relation_terms):
    answer_hit, matched_answer = text_contains_any_term(evidence_text, answer_terms)
    relation_hit, matched_relation = text_contains_any_term(evidence_text, relation_terms)

    if answer_hit and relation_hit:
        strength = "STRONG_ANSWER_AND_RELATION_MATCH"
    elif answer_hit:
        strength = "ANSWER_ONLY_MATCH"
    elif relation_hit:
        strength = "RELATION_ONLY_MATCH"
    else:
        strength = "NO_EVIDENCE"

    return {
        "evidence_strength": strength,
        "answer_grounded_in_evidence": bool(answer_hit),
        "relation_grounded_in_evidence": bool(relation_hit),
        "matched_answer_term": matched_answer,
        "matched_relation_keyword": matched_relation
    }


def retrieve_evidence_for_row(row):
    candidates = retrieve_subject_page_candidates(
        row,
        bm25_k=BM25_CANDIDATES,
        dense_k=DENSE_CANDIDATES
    )

    if len(candidates) == 0:
        return [], "no_subject_page_evidence"

    reranked = rerank_candidates(
        row["claim"],
        candidates,
        max_candidates=RERANK_CANDIDATES
    )

    scored = []

    for c in reranked:
        grounding = classify_evidence_strength(
            c["text"],
            row["answer_grounding_terms"],
            row["relation_grounding_terms"]
        )

        c.update(grounding)

        # Keep useful evidence:
        # - answer evidence can support missing facts
        # - relation-only evidence can contradict wrong answers
        if c["evidence_strength"] != "NO_EVIDENCE":
            scored.append(c)

    if len(scored) == 0:
        return [], "no_filtered_evidence"

    return scored[:TOP_EVIDENCE_PER_CLAIM], "subject_page_filtered"


# ============================================================
# 12. Retrieve evidence for verification rows
# ============================================================

retrieved_rows = []

for _, row in tqdm(verify_df.iterrows(), total=len(verify_df)):
    results, retrieval_scope = retrieve_evidence_for_row(row)

    base_row = {
        "row_index": int(row["row_index"]),  # ORIGINAL Stage 3 row index
        "verification_row_index": int(row["verification_row_index"]),
        "id": row["id"],
        "question": row["question"],
        "subject_label": row["subject_label"],
        "relation_label": row["relation_label"],
        "relation_normalized": row["relation_normalized"],
        "reference_answer": row["reference_answer"],
        "popqa_possible_answers": row["popqa_possible_answers"],
        "model_raw_answer": row["model_raw_answer"],
        "model_extracted_answer": row["model_extracted_answer"],
        "model_clean_answer": row["model_clean_answer"],
        "claim": row["claim"],
        "model_answer_matches_reference": bool(row["model_answer_matches_reference"]),
        "answer_case": row["answer_case"],
        "stage3_answer_case": row["stage3_answer_case"],
        "kg_fact_status_checked": row["kg_fact_status_checked"],
        "kg_only_prediction": row["kg_only_prediction"],
        "subject_popularity_bin": row["subject_popularity_bin"],
        "s_wiki_title": row["s_wiki_title"],
        "answer_grounding_terms": json.dumps(row["answer_grounding_terms"], ensure_ascii=False),
        "relation_grounding_terms": json.dumps(row["relation_grounding_terms"], ensure_ascii=False),
    }

    if len(results) == 0:
        retrieved_rows.append({
            **base_row,
            "evidence_rank": 0,
            "evidence_doc_id": "",
            "evidence_wiki_title": "",
            "evidence_text": "",
            "bm25_score": 0.0,
            "dense_score": 0.0,
            "reranker_score": 0.0,
            "retrieval_scope": retrieval_scope,
            "evidence_strength": "NO_EVIDENCE",
            "answer_grounded_in_evidence": False,
            "relation_grounded_in_evidence": False,
            "matched_answer_term": "",
            "matched_relation_keyword": ""
        })
        continue

    for rank, result in enumerate(results, start=1):
        retrieved_rows.append({
            **base_row,
            "evidence_rank": rank,
            "evidence_doc_id": result["doc_id"],
            "evidence_wiki_title": result["wiki_title"],
            "evidence_text": result["text"],
            "bm25_score": result["bm25_score"],
            "dense_score": result["dense_score"],
            "reranker_score": result["reranker_score"],
            "retrieval_scope": retrieval_scope,
            "evidence_strength": result["evidence_strength"],
            "answer_grounded_in_evidence": bool(result["answer_grounded_in_evidence"]),
            "relation_grounded_in_evidence": bool(result["relation_grounded_in_evidence"]),
            "matched_answer_term": result["matched_answer_term"],
            "matched_relation_keyword": result["matched_relation_keyword"]
        })

retrieved_evidence_df = pd.DataFrame(retrieved_rows)


# ============================================================
# 13. Safety checks for row alignment
# ============================================================

if retrieved_evidence_df["row_index"].min() < 0:
    raise ValueError("Invalid row_index: contains negative values.")

if retrieved_evidence_df["row_index"].max() >= len(stage3_df):
    raise ValueError("Invalid row_index: exceeds Stage 3 row count.")

alignment_check = retrieved_evidence_df.merge(
    stage3_df[["row_index", "id", "question"]],
    on="row_index",
    how="left",
    suffixes=("", "_stage3")
)

id_mismatch_count = int(
    (alignment_check["id"].astype(str) != alignment_check["id_stage3"].astype(str)).sum()
)

question_mismatch_count = int(
    (alignment_check["question"].astype(str) != alignment_check["question_stage3"].astype(str)).sum()
)

if id_mismatch_count != 0 or question_mismatch_count != 0:
    raise ValueError(
        f"Stage 4 row alignment failed. "
        f"ID mismatches: {id_mismatch_count}, "
        f"question mismatches: {question_mismatch_count}"
    )

print("\nRow alignment check passed.")
print("ID mismatch count:", id_mismatch_count)
print("Question mismatch count:", question_mismatch_count)


# ============================================================
# 14. Save Stage 4 outputs
# ============================================================

retrieved_evidence_df.to_csv(STAGE4_OUTPUT_PATH, index=False)

print("\nSaved Stage 4 evidence:", STAGE4_OUTPUT_PATH)

print("\nRetrieved evidence rows:", len(retrieved_evidence_df))
print("Unique original Stage 3 rows with evidence records:", retrieved_evidence_df["row_index"].nunique())
print("row_index min:", int(retrieved_evidence_df["row_index"].min()))
print("row_index max:", int(retrieved_evidence_df["row_index"].max()))

print("\nRetrieval scope counts:")
print(retrieved_evidence_df["retrieval_scope"].value_counts())

print("\nEvidence strength counts:")
print(retrieved_evidence_df["evidence_strength"].value_counts())

print("\nAnswer-grounded evidence counts:")
print(retrieved_evidence_df["answer_grounded_in_evidence"].value_counts())

print("\nRelation-grounded evidence counts:")
print(retrieved_evidence_df["relation_grounded_in_evidence"].value_counts())

print("\nClean evidence preview:")
display(retrieved_evidence_df[[
    "row_index",
    "verification_row_index",
    "claim",
    "stage3_answer_case",
    "evidence_rank",
    "evidence_wiki_title",
    "retrieval_scope",
    "evidence_strength",
    "answer_grounded_in_evidence",
    "relation_grounded_in_evidence",
    "matched_answer_term",
    "matched_relation_keyword",
    "reranker_score",
    "evidence_text"
]].head(25))


# ============================================================
# 15. Save metadata
# ============================================================

def value_counts_dict(series):
    return {str(k): int(v) for k, v in series.value_counts().to_dict().items()}


metadata = {
    "stage": "Stage 4 Final Clean Wikipedia Evidence Retrieval",
    "input_file": STAGE3_PATH,
    "output_file": STAGE4_OUTPUT_PATH,
    "cache_file": CACHE_PATH,
    "evidence_corpus_file": EVIDENCE_CORPUS_PATH,

    "num_stage3_rows": int(len(stage3_df)),
    "num_rows_needing_verification": int(len(verify_df)),
    "num_unique_subject_wikipedia_titles": int(len(unique_titles)),
    "num_evidence_chunks": int(len(evidence_corpus)),
    "num_retrieved_evidence_rows": int(len(retrieved_evidence_df)),

    "embedding_model": EMBEDDING_MODEL_NAME,
    "reranker_model": RERANKER_MODEL_NAME,
    "retrieval_method": (
        "PopQA subject Wikipedia page retrieval using BM25 + BGE dense retrieval "
        "+ BGE reranker, followed by answer/relation grounding filters."
    ),

    "bm25_candidates": int(BM25_CANDIDATES),
    "dense_candidates": int(DENSE_CANDIDATES),
    "rerank_candidates": int(RERANK_CANDIDATES),
    "top_evidence_per_claim": int(TOP_EVIDENCE_PER_CLAIM),

    "row_index_policy": {
        "row_index": "Original row index from the 500-row Stage 3 file.",
        "verification_row_index": "Filtered verification-row index used only for audit/debugging."
    },

    "row_alignment_check": {
        "id_mismatch_count": int(id_mismatch_count),
        "question_mismatch_count": int(question_mismatch_count),
        "row_index_min": int(retrieved_evidence_df["row_index"].min()),
        "row_index_max": int(retrieved_evidence_df["row_index"].max()),
        "unique_original_stage3_rows": int(retrieved_evidence_df["row_index"].nunique())
    },

    "retrieval_scope_counts": value_counts_dict(retrieved_evidence_df["retrieval_scope"]),
    "evidence_strength_counts": value_counts_dict(retrieved_evidence_df["evidence_strength"]),
    "answer_grounded_counts": value_counts_dict(retrieved_evidence_df["answer_grounded_in_evidence"]),
    "relation_grounded_counts": value_counts_dict(retrieved_evidence_df["relation_grounded_in_evidence"]),

    "stage3_answer_case_counts_in_verification_rows": value_counts_dict(verify_df["stage3_answer_case"]),
    "kg_only_prediction_counts": value_counts_dict(stage3_df["kg_only_prediction"]),

    "important_design_choices": {
        "uses_wikipedia_only": True,
        "uses_popqa_subject_wiki_title": True,
        "does_not_search_broad_wikipedia_by_subject_label": True,
        "manual_answer_alias_map_used": False,
        "wikidata_aliases_used": False,
        "popqa_possible_answers_used_for_grounding_only_when_model_answer_matches_reference": True,
        "keeps_relation_only_evidence": True,
        "why_keep_relation_only_evidence": (
            "Relation-only chunks often contain the true relation value and are useful for NLI contradiction detection."
        )
    },

    "label_definitions": {
        "retrieval_scope": {
            "subject_page_filtered": "Evidence was retrieved from the subject Wikipedia page and passed grounding filters.",
            "no_filtered_evidence": "Subject page existed, but no candidate chunk passed grounding filters.",
            "no_subject_page_evidence": "No usable subject Wikipedia page evidence was available."
        },
        "evidence_strength": {
            "STRONG_ANSWER_AND_RELATION_MATCH": "Evidence mentions the answer and a relation clue.",
            "ANSWER_ONLY_MATCH": "Evidence mentions the answer but not a relation clue.",
            "RELATION_ONLY_MATCH": "Evidence mentions a relation clue but not the model answer. Useful for contradiction.",
            "NO_EVIDENCE": "No acceptable evidence chunk was found."
        }
    }
}

with open(STAGE4_METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved Stage 4 metadata:", STAGE4_METADATA_PATH)

print("\nStage 4 complete.")
# Stage 5
# ============================================================
# STAGE 5 FINAL:
# NLI evidence verification + safer hybrid hallucination detector
#
# Inputs:
# - data_final/stage3_kg_only_predictions.csv
# - data_final/stage4_claims_with_evidence.csv
#
# Outputs:
# - data_final/stage5_nli_evidence_predictions.csv
# - data_final/stage5_hybrid_predictions.csv
# - data_final/stage5_hybrid_metrics.json
# ============================================================

!pip install -q transformers accelerate sentencepiece pandas numpy tqdm scikit-learn

import os
import json
import gc
import numpy as np
import pandas as pd
import torch

from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# 0. Free Stage 4 model memory if still loaded
# ============================================================

for var_name in [
    "embedding_model",
    "reranker_model",
    "corpus_embeddings",
    "evidence_corpus",
    "corpus_texts",
    "bm25_index"
]:
    try:
        del globals()[var_name]
    except Exception:
        pass

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# 1. Config
# ============================================================

DATA_DIR = "data_final"

STAGE3_CANDIDATES = [
    f"{DATA_DIR}/stage3_kg_only_predictions.csv",
    f"{DATA_DIR}/stage3_kg_only_genai_predictions.csv"
]

STAGE4_CANDIDATES = [
    f"{DATA_DIR}/stage4_claims_with_evidence.csv",
    f"{DATA_DIR}/stage4_genai_claims_with_evidence.csv"
]

STAGE5_EVIDENCE_OUTPUT = f"{DATA_DIR}/stage5_nli_evidence_predictions.csv"
STAGE5_FINAL_OUTPUT = f"{DATA_DIR}/stage5_hybrid_predictions.csv"
STAGE5_METRICS_OUTPUT = f"{DATA_DIR}/stage5_hybrid_metrics.json"

NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# Centralized labels
KG_ONLY_SUPPORTED = "SUPPORTED_BY_KG_ONLY"
KG_ONLY_FLAGGED = "FLAGGED_AS_HALLUCINATION_BY_KG_ONLY"
UNUSABLE_MODEL_ANSWER = "UNUSABLE_MODEL_ANSWER"

ANSWER_CORRECT_PRESENT = "CORRECT_AND_FACT_PRESENT_IN_KG"
ANSWER_CORRECT_MISSING = "CORRECT_BUT_FACT_MISSING_FROM_KG"
ANSWER_WRONG = "WRONG_MODEL_ANSWER"

FACT_PRESENT = "FACT_PRESENT_IN_KG"
FACT_MISSING = "FACT_MISSING_FROM_KG"

EVIDENCE_SUPPORTED = "SUPPORTED_BY_EVIDENCE"
EVIDENCE_CONTRADICTED = "CONTRADICTED_BY_EVIDENCE"
EVIDENCE_NOT_ENOUGH = "NOT_ENOUGH_EVIDENCE"

HYBRID_SUPPORTED_BY_KG = "SUPPORTED_BY_KG"
HYBRID_VERIFIED_BY_TEXT = "VERIFIED_BY_TEXT_EVIDENCE"
HYBRID_CONTRADICTED_BY_TEXT = "CONTRADICTED_BY_TEXT_EVIDENCE"
HYBRID_NOT_ENOUGH = "NOT_ENOUGH_EVIDENCE"

NOT_HALLUCINATION = "NOT_HALLUCINATION"
HALLUCINATION = "HALLUCINATION"

# Conservative NLI thresholds
THRESHOLDS = {
    "support_entailment_min": 0.85,
    "support_contradiction_max": 0.20,
    "contradiction_min": 0.75,
    "contradiction_entailment_max": 0.25,
    "support_margin": 0.15,
    "contradiction_margin": 0.10,
    "strict_contradiction_min": 0.90
}

# Evidence-strength policy from Stage 4
SUPPORT_ALLOWED_STRENGTHS = {
    "STRONG_ANSWER_AND_RELATION_MATCH",
    "ANSWER_ONLY_MATCH"
}

CONTRADICTION_ALLOWED_STRENGTHS = {
    "STRONG_ANSWER_AND_RELATION_MATCH",
    "RELATION_ONLY_MATCH"
}

VALID_RETRIEVAL_SCOPES = {
    "subject_page_filtered",
    "subject_page"
}


def first_existing_path(paths, label):
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No valid {label} file found. Checked: {paths}")


STAGE3_PATH = first_existing_path(STAGE3_CANDIDATES, "Stage 3")
STAGE4_PATH = first_existing_path(STAGE4_CANDIDATES, "Stage 4")

print("Using Stage 3 file:", STAGE3_PATH)
print("Using Stage 4 file:", STAGE4_PATH)


# ============================================================
# 2. Load files
# ============================================================

stage3_df = pd.read_csv(STAGE3_PATH)
evidence_df = pd.read_csv(STAGE4_PATH)

# Make row_index mean original Stage 3 row position.
# This is needed because Stage 4 row_index points back to Stage 3.
if "row_index" in stage3_df.columns:
    stage3_df = stage3_df.drop(columns=["row_index"])

stage3_df = stage3_df.reset_index(drop=True)
stage3_df.insert(0, "row_index", stage3_df.index)

print("\nStage 3 rows:", len(stage3_df))
print("Stage 4 evidence rows:", len(evidence_df))

required_stage3_cols = [
    "row_index",
    "id",
    "question",
    "reference_answer",
    "model_raw_answer",
    "model_extracted_answer",
    "model_answer_matches_reference",
    "kg_only_prediction",
    "stage3_answer_case",
    "kg_fact_status_checked",
    "relation_normalized",
    "subject_popularity_bin"
]

required_stage4_cols = [
    "row_index",
    "claim",
    "evidence_text",
    "evidence_rank",
    "retrieval_scope",
    "evidence_strength",
    "answer_grounded_in_evidence",
    "relation_grounded_in_evidence",
    "reranker_score"
]

missing_stage3 = [c for c in required_stage3_cols if c not in stage3_df.columns]
missing_stage4 = [c for c in required_stage4_cols if c not in evidence_df.columns]

if missing_stage3:
    raise ValueError(f"Stage 3 is missing required columns: {missing_stage3}")

if missing_stage4:
    raise ValueError(f"Stage 4 is missing required columns: {missing_stage4}")


# ============================================================
# 3. Helper conversion functions
# ============================================================

def to_bool(value):
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    text = str(value).strip().lower()

    return text in {"true", "1", "yes", "y"}


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


# Normalize key boolean/numeric columns
stage3_df["model_answer_matches_reference"] = stage3_df["model_answer_matches_reference"].apply(to_bool)

evidence_df["answer_grounded_in_evidence"] = evidence_df["answer_grounded_in_evidence"].apply(to_bool)
evidence_df["relation_grounded_in_evidence"] = evidence_df["relation_grounded_in_evidence"].apply(to_bool)
evidence_df["evidence_rank"] = evidence_df["evidence_rank"].apply(safe_int)
evidence_df["reranker_score"] = evidence_df["reranker_score"].apply(safe_float)

evidence_df["claim"] = evidence_df["claim"].fillna("").astype(str)
evidence_df["evidence_text"] = evidence_df["evidence_text"].fillna("").astype(str)
evidence_df["retrieval_scope"] = evidence_df["retrieval_scope"].fillna("").astype(str)
evidence_df["evidence_strength"] = evidence_df["evidence_strength"].fillna("NO_EVIDENCE").astype(str)

print("\nKG-only prediction distribution:")
print(stage3_df["kg_only_prediction"].value_counts())

print("\nStage 3 answer case distribution:")
print(stage3_df["stage3_answer_case"].value_counts())

print("\nKG fact status distribution:")
print(stage3_df["kg_fact_status_checked"].value_counts())

print("\nStage 4 retrieval scope distribution:")
print(evidence_df["retrieval_scope"].value_counts())

print("\nStage 4 evidence strength distribution:")
print(evidence_df["evidence_strength"].value_counts())


# ============================================================
# 4. Row alignment check
# ============================================================

alignment_cols = ["row_index", "id", "question"]

stage3_alignment = stage3_df[alignment_cols].copy()

stage4_alignment = (
    evidence_df[alignment_cols]
    .drop_duplicates(subset=["row_index"])
    .copy()
)

alignment_check = stage4_alignment.merge(
    stage3_alignment,
    on="row_index",
    how="left",
    suffixes=("_stage4", "_stage3")
)

id_mismatch = (
    alignment_check["id_stage4"].astype(str) !=
    alignment_check["id_stage3"].astype(str)
).sum()

question_mismatch = (
    alignment_check["question_stage4"].astype(str) !=
    alignment_check["question_stage3"].astype(str)
).sum()

print("\nRow alignment check:")
print("Stage 4 unique row_index values:", evidence_df["row_index"].nunique())
print("ID mismatch count:", int(id_mismatch))
print("Question mismatch count:", int(question_mismatch))

if id_mismatch > 0 or question_mismatch > 0:
    raise ValueError(
        "Stage 4 row_index does not align with Stage 3. "
        "Do not continue until Stage 4 is regenerated correctly."
    )


# ============================================================
# 5. Prepare evidence rows for NLI
# ============================================================

valid_evidence_df = evidence_df[
    (evidence_df["claim"].str.strip() != "") &
    (evidence_df["evidence_text"].str.strip() != "") &
    (evidence_df["evidence_rank"] > 0) &
    (evidence_df["retrieval_scope"].isin(VALID_RETRIEVAL_SCOPES)) &
    (evidence_df["evidence_strength"] != "NO_EVIDENCE")
].copy().reset_index(drop=True)

print("\nValid evidence rows for NLI:", len(valid_evidence_df))

preview_cols = [
    "row_index",
    "claim",
    "stage3_answer_case",
    "evidence_rank",
    "evidence_wiki_title",
    "retrieval_scope",
    "evidence_strength",
    "answer_grounded_in_evidence",
    "relation_grounded_in_evidence",
    "reranker_score",
    "evidence_text"
]

display(valid_evidence_df[preview_cols].head(15))


# ============================================================
# 6. Load NLI model
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("\nDevice:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)

nli_model.to(device)
nli_model.eval()

id2label = nli_model.config.id2label

def normalize_nli_label(label):
    label = str(label).lower()

    if "entail" in label:
        return "entailment"

    if "contrad" in label:
        return "contradiction"

    if "neutral" in label:
        return "neutral"

    return label


label_map = {
    idx: normalize_nli_label(label)
    for idx, label in id2label.items()
}

print("\nNLI model:", NLI_MODEL_NAME)
print("Raw model labels:", id2label)
print("Normalized label map:", label_map)


# ============================================================
# 7. Run evidence-level NLI
# ============================================================

def run_nli_batch(premises, hypotheses, batch_size=16, max_length=384):
    rows = []

    for start in tqdm(range(0, len(premises), batch_size)):
        batch_premises = premises[start:start + batch_size]
        batch_hypotheses = hypotheses[start:start + batch_size]

        inputs = tokenizer(
            batch_premises,
            batch_hypotheses,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = nli_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()

        for prob_row in probs:
            scores = {
                label_map[i]: float(prob_row[i])
                for i in range(len(prob_row))
            }

            entailment = scores.get("entailment", 0.0)
            neutral = scores.get("neutral", 0.0)
            contradiction = scores.get("contradiction", 0.0)

            score_dict = {
                "entailment": entailment,
                "neutral": neutral,
                "contradiction": contradiction
            }

            rows.append({
                "nli_pred_label": max(score_dict, key=score_dict.get),
                "nli_entailment_score": entailment,
                "nli_neutral_score": neutral,
                "nli_contradiction_score": contradiction
            })

    return pd.DataFrame(rows)


if len(valid_evidence_df) > 0:
    BATCH_SIZE = 16 if device == "cuda" else 4

    nli_outputs = run_nli_batch(
        valid_evidence_df["evidence_text"].tolist(),
        valid_evidence_df["claim"].tolist(),
        batch_size=BATCH_SIZE,
        max_length=384
    )

    evidence_nli_df = pd.concat(
        [valid_evidence_df.reset_index(drop=True), nli_outputs.reset_index(drop=True)],
        axis=1
    )

else:
    evidence_nli_df = valid_evidence_df.copy()

    for col in [
        "nli_pred_label",
        "nli_entailment_score",
        "nli_neutral_score",
        "nli_contradiction_score"
    ]:
        evidence_nli_df[col] = []

evidence_nli_df.to_csv(STAGE5_EVIDENCE_OUTPUT, index=False)

print("\nSaved evidence-level NLI predictions:", STAGE5_EVIDENCE_OUTPUT)

display(evidence_nli_df[[
    "row_index",
    "claim",
    "stage3_answer_case",
    "evidence_rank",
    "evidence_strength",
    "answer_grounded_in_evidence",
    "relation_grounded_in_evidence",
    "nli_pred_label",
    "nli_entailment_score",
    "nli_contradiction_score",
    "nli_neutral_score",
    "evidence_text"
]].head(20))


# ============================================================
# 8. Safer aggregation per claim / Stage 3 row
# ============================================================

def aggregate_nli_group(group):
    """
    Key safety rules:
    - Text support requires answer-grounded evidence.
    - RELATION_ONLY_MATCH cannot support a claim.
    - RELATION_ONLY_MATCH can contradict a claim.
    - NO_EVIDENCE never enters this function.
    """

    group = group.copy()

    group["support_allowed_by_grounding"] = (
        group["answer_grounded_in_evidence"] &
        group["evidence_strength"].isin(SUPPORT_ALLOWED_STRENGTHS)
    )

    group["contradiction_allowed_by_grounding"] = (
        group["relation_grounded_in_evidence"] &
        group["evidence_strength"].isin(CONTRADICTION_ALLOWED_STRENGTHS)
    )

    group["support_candidate_score"] = np.where(
        (
            group["support_allowed_by_grounding"] &
            (group["nli_entailment_score"] >= THRESHOLDS["support_entailment_min"]) &
            (group["nli_contradiction_score"] <= THRESHOLDS["support_contradiction_max"])
        ),
        group["nli_entailment_score"],
        0.0
    )

    group["contradiction_candidate_score"] = np.where(
        (
            group["contradiction_allowed_by_grounding"] &
            (group["nli_contradiction_score"] >= THRESHOLDS["contradiction_min"]) &
            (group["nli_entailment_score"] <= THRESHOLDS["contradiction_entailment_max"])
        ),
        group["nli_contradiction_score"],
        0.0
    )

    best_support_score = float(group["support_candidate_score"].max())
    best_contradiction_score = float(group["contradiction_candidate_score"].max())

    max_entailment_score = float(group["nli_entailment_score"].max())
    max_contradiction_score = float(group["nli_contradiction_score"].max())
    max_neutral_score = float(group["nli_neutral_score"].max())

    if best_support_score > 0:
        best_support_row = group.loc[group["support_candidate_score"].idxmax()]
    else:
        best_support_row = group.loc[group["nli_entailment_score"].idxmax()]

    if best_contradiction_score > 0:
        best_contradiction_row = group.loc[group["contradiction_candidate_score"].idxmax()]
    else:
        best_contradiction_row = group.loc[group["nli_contradiction_score"].idxmax()]

    support_wins = (
        best_support_score >= THRESHOLDS["support_entailment_min"] and
        best_support_score >= best_contradiction_score + THRESHOLDS["support_margin"]
    )

    contradiction_wins = (
        best_contradiction_score >= THRESHOLDS["contradiction_min"] and
        best_contradiction_score >= best_support_score + THRESHOLDS["contradiction_margin"]
    )

    strict_contradiction_wins = (
        best_contradiction_score >= THRESHOLDS["strict_contradiction_min"] and
        best_support_score < THRESHOLDS["support_entailment_min"]
    )

    if support_wins:
        verdict = EVIDENCE_SUPPORTED
        best_row = best_support_row

    elif contradiction_wins or strict_contradiction_wins:
        verdict = EVIDENCE_CONTRADICTED
        best_row = best_contradiction_row

    else:
        verdict = EVIDENCE_NOT_ENOUGH

        group["max_nli_confidence"] = group[[
            "nli_entailment_score",
            "nli_neutral_score",
            "nli_contradiction_score"
        ]].max(axis=1)

        best_row = group.loc[group["max_nli_confidence"].idxmax()]

    return {
        "row_index": int(best_row["row_index"]),
        "id": best_row["id"],
        "question": best_row["question"],
        "claim": best_row["claim"],
        "evidence_verdict": verdict,

        "best_support_score": best_support_score,
        "best_contradiction_score": best_contradiction_score,
        "max_entailment_score": max_entailment_score,
        "max_contradiction_score": max_contradiction_score,
        "max_neutral_score": max_neutral_score,

        "best_evidence_rank": int(best_row["evidence_rank"]),
        "best_evidence_wiki_title": best_row.get("evidence_wiki_title", ""),
        "best_evidence_text": best_row["evidence_text"],
        "best_reranker_score": float(best_row["reranker_score"]),

        "best_evidence_strength": best_row["evidence_strength"],
        "best_answer_grounded_in_evidence": bool(best_row["answer_grounded_in_evidence"]),
        "best_relation_grounded_in_evidence": bool(best_row["relation_grounded_in_evidence"]),
        "best_matched_answer_term": best_row.get("matched_answer_term", ""),
        "best_matched_relation_keyword": best_row.get("matched_relation_keyword", ""),

        "best_nli_label": best_row["nli_pred_label"],
        "best_nli_entailment_score": float(best_row["nli_entailment_score"]),
        "best_nli_contradiction_score": float(best_row["nli_contradiction_score"]),
        "best_nli_neutral_score": float(best_row["nli_neutral_score"]),

        "num_evidence_rows_used": int(len(group)),
        "num_support_candidates": int((group["support_candidate_score"] > 0).sum()),
        "num_contradiction_candidates": int((group["contradiction_candidate_score"] > 0).sum())
    }


aggregated_rows = []

for row_index, group in evidence_nli_df.groupby("row_index"):
    aggregated_rows.append(aggregate_nli_group(group))

aggregated_nli_df = pd.DataFrame(aggregated_rows)

print("\nAggregated NLI rows:", len(aggregated_nli_df))

if len(aggregated_nli_df) > 0:
    print("\nEvidence verdict distribution:")
    print(aggregated_nli_df["evidence_verdict"].value_counts())

    display(aggregated_nli_df[[
        "row_index",
        "question",
        "claim",
        "evidence_verdict",
        "best_support_score",
        "best_contradiction_score",
        "best_evidence_strength",
        "best_answer_grounded_in_evidence",
        "best_relation_grounded_in_evidence",
        "best_evidence_text"
    ]].head(20))


# ============================================================
# 9. Build final hybrid predictions
# ============================================================

final_df = stage3_df.copy()

# Merge aggregated NLI verdicts back to full Stage 3 rows
if len(aggregated_nli_df) > 0:
    final_df = final_df.merge(
        aggregated_nli_df,
        on=["row_index", "id", "question"],
        how="left"
    )
else:
    for col in [
        "claim",
        "evidence_verdict",
        "best_support_score",
        "best_contradiction_score",
        "max_entailment_score",
        "max_contradiction_score",
        "max_neutral_score",
        "best_evidence_rank",
        "best_evidence_wiki_title",
        "best_evidence_text",
        "best_reranker_score",
        "best_evidence_strength",
        "best_answer_grounded_in_evidence",
        "best_relation_grounded_in_evidence",
        "best_matched_answer_term",
        "best_matched_relation_keyword",
        "best_nli_label",
        "best_nli_entailment_score",
        "best_nli_contradiction_score",
        "best_nli_neutral_score",
        "num_evidence_rows_used",
        "num_support_candidates",
        "num_contradiction_candidates"
    ]:
        final_df[col] = np.nan

final_df["evidence_verdict"] = final_df["evidence_verdict"].fillna(EVIDENCE_NOT_ENOUGH)

# Default prediction
final_df["hybrid_final_label"] = HYBRID_NOT_ENOUGH

# KG-supported rows stay supported
final_df.loc[
    final_df["kg_only_prediction"] == KG_ONLY_SUPPORTED,
    "hybrid_final_label"
] = HYBRID_SUPPORTED_BY_KG

# Unusable rows stay unusable
final_df.loc[
    final_df["kg_only_prediction"] == UNUSABLE_MODEL_ANSWER,
    "hybrid_final_label"
] = UNUSABLE_MODEL_ANSWER

# Evidence-verified rows
flagged_mask = final_df["kg_only_prediction"] == KG_ONLY_FLAGGED

final_df.loc[
    flagged_mask & (final_df["evidence_verdict"] == EVIDENCE_SUPPORTED),
    "hybrid_final_label"
] = HYBRID_VERIFIED_BY_TEXT

final_df.loc[
    flagged_mask & (final_df["evidence_verdict"] == EVIDENCE_CONTRADICTED),
    "hybrid_final_label"
] = HYBRID_CONTRADICTED_BY_TEXT

print("\nHybrid final label distribution:")
print(final_df["hybrid_final_label"].value_counts())

display(final_df[[
    "row_index",
    "question",
    "reference_answer",
    "model_raw_answer",
    "model_extracted_answer",
    "stage3_answer_case",
    "kg_only_prediction",
    "hybrid_final_label",
    "evidence_verdict",
    "best_support_score",
    "best_contradiction_score",
    "best_evidence_strength",
    "best_answer_grounded_in_evidence",
    "best_relation_grounded_in_evidence",
    "best_evidence_text"
]].head(30))


# ============================================================
# 10. Evaluation
# ============================================================

eval_df = final_df[
    final_df["stage3_answer_case"] != UNUSABLE_MODEL_ANSWER
].copy()

eval_df["reference_binary_label"] = np.where(
    eval_df["model_answer_matches_reference"],
    NOT_HALLUCINATION,
    HALLUCINATION
)

eval_df["kg_only_binary_prediction"] = np.where(
    eval_df["kg_only_prediction"] == KG_ONLY_SUPPORTED,
    NOT_HALLUCINATION,
    HALLUCINATION
)

eval_df["hybrid_binary_prediction"] = np.where(
    eval_df["hybrid_final_label"].isin([
        HYBRID_SUPPORTED_BY_KG,
        HYBRID_VERIFIED_BY_TEXT
    ]),
    NOT_HALLUCINATION,
    HALLUCINATION
)

labels = [NOT_HALLUCINATION, HALLUCINATION]

kg_only_accuracy = accuracy_score(
    eval_df["reference_binary_label"],
    eval_df["kg_only_binary_prediction"]
)

hybrid_accuracy = accuracy_score(
    eval_df["reference_binary_label"],
    eval_df["hybrid_binary_prediction"]
)

true_not_hallucination = eval_df["reference_binary_label"] == NOT_HALLUCINATION
true_hallucination = eval_df["reference_binary_label"] == HALLUCINATION

kg_false_hallucinations = eval_df[
    true_not_hallucination &
    (eval_df["kg_only_binary_prediction"] == HALLUCINATION)
].copy()

hybrid_false_hallucinations = eval_df[
    true_not_hallucination &
    (eval_df["hybrid_binary_prediction"] == HALLUCINATION)
].copy()

kg_false_hallucination_rate = (
    len(kg_false_hallucinations) / int(true_not_hallucination.sum())
    if int(true_not_hallucination.sum()) > 0
    else 0.0
)

hybrid_false_hallucination_rate = (
    len(hybrid_false_hallucinations) / int(true_not_hallucination.sum())
    if int(true_not_hallucination.sum()) > 0
    else 0.0
)

missing_fact_df = eval_df[
    eval_df["stage3_answer_case"] == ANSWER_CORRECT_MISSING
].copy()

missing_fact_recovery_rate = (
    (missing_fact_df["hybrid_final_label"] == HYBRID_VERIFIED_BY_TEXT).mean()
    if len(missing_fact_df) > 0
    else 0.0
)

false_acceptances = eval_df[
    true_hallucination &
    (eval_df["hybrid_binary_prediction"] == NOT_HALLUCINATION)
].copy()

false_acceptance_rate = (
    len(false_acceptances) / int(true_hallucination.sum())
    if int(true_hallucination.sum()) > 0
    else 0.0
)

hybrid_cm = confusion_matrix(
    eval_df["reference_binary_label"],
    eval_df["hybrid_binary_prediction"],
    labels=labels
)

kg_cm = confusion_matrix(
    eval_df["reference_binary_label"],
    eval_df["kg_only_binary_prediction"],
    labels=labels
)

print("\n================ STAGE 5 FINAL HYBRID METRICS ================")
print("Evaluated rows:", len(eval_df))

print("\nKG-only accuracy:", round(float(kg_only_accuracy), 4))
print("Hybrid accuracy:", round(float(hybrid_accuracy), 4))

print("\nCorrect model answers:", int(true_not_hallucination.sum()))
print("KG-only false hallucinations:", len(kg_false_hallucinations))
print("Hybrid false hallucinations:", len(hybrid_false_hallucinations))

print("\nKG-only false hallucination rate:", round(float(kg_false_hallucination_rate), 4))
print("Hybrid false hallucination rate:", round(float(hybrid_false_hallucination_rate), 4))

print("\nCorrect-but-fact-missing cases:", len(missing_fact_df))
print("Missing fact recovery rate:", round(float(missing_fact_recovery_rate), 4))

print("\nFalse acceptances introduced by hybrid:", len(false_acceptances))
print("False acceptance rate:", round(float(false_acceptance_rate), 4))

print("\nHybrid confusion matrix:")
display(pd.DataFrame(
    hybrid_cm,
    index=[f"true_{x}" for x in labels],
    columns=[f"pred_{x}" for x in labels]
))

print("\nHybrid classification report:")
print(classification_report(
    eval_df["reference_binary_label"],
    eval_df["hybrid_binary_prediction"],
    labels=labels,
    zero_division=0
))


# ============================================================
# 11. Relation and popularity breakdown
# ============================================================

relation_rows = []

for relation, group in eval_df.groupby("relation_normalized"):
    group_true_correct = group["reference_binary_label"] == NOT_HALLUCINATION

    group_kg_false = group[
        group_true_correct &
        (group["kg_only_binary_prediction"] == HALLUCINATION)
    ]

    group_hybrid_false = group[
        group_true_correct &
        (group["hybrid_binary_prediction"] == HALLUCINATION)
    ]

    group_missing = group[group["stage3_answer_case"] == ANSWER_CORRECT_MISSING]

    relation_rows.append({
        "relation": relation,
        "num_rows": int(len(group)),
        "correct_model_answers": int(group_true_correct.sum()),
        "kg_only_false_hallucinations": int(len(group_kg_false)),
        "hybrid_false_hallucinations": int(len(group_hybrid_false)),
        "kg_only_false_hallucination_rate": float(
            len(group_kg_false) / int(group_true_correct.sum())
            if int(group_true_correct.sum()) > 0 else 0.0
        ),
        "hybrid_false_hallucination_rate": float(
            len(group_hybrid_false) / int(group_true_correct.sum())
            if int(group_true_correct.sum()) > 0 else 0.0
        ),
        "correct_but_fact_missing_cases": int(len(group_missing)),
        "missing_fact_recovery_rate": float(
            (group_missing["hybrid_final_label"] == HYBRID_VERIFIED_BY_TEXT).mean()
            if len(group_missing) > 0 else 0.0
        )
    })

relation_breakdown = pd.DataFrame(relation_rows).sort_values(
    "kg_only_false_hallucination_rate",
    ascending=False
)

print("\nRelation-level breakdown:")
display(relation_breakdown)


popularity_rows = []

for pop_bin, group in eval_df.groupby("subject_popularity_bin"):
    group_true_correct = group["reference_binary_label"] == NOT_HALLUCINATION

    group_kg_false = group[
        group_true_correct &
        (group["kg_only_binary_prediction"] == HALLUCINATION)
    ]

    group_hybrid_false = group[
        group_true_correct &
        (group["hybrid_binary_prediction"] == HALLUCINATION)
    ]

    group_missing = group[group["stage3_answer_case"] == ANSWER_CORRECT_MISSING]

    popularity_rows.append({
        "subject_popularity_bin": pop_bin,
        "num_rows": int(len(group)),
        "correct_model_answers": int(group_true_correct.sum()),
        "kg_only_false_hallucinations": int(len(group_kg_false)),
        "hybrid_false_hallucinations": int(len(group_hybrid_false)),
        "kg_only_false_hallucination_rate": float(
            len(group_kg_false) / int(group_true_correct.sum())
            if int(group_true_correct.sum()) > 0 else 0.0
        ),
        "hybrid_false_hallucination_rate": float(
            len(group_hybrid_false) / int(group_true_correct.sum())
            if int(group_true_correct.sum()) > 0 else 0.0
        ),
        "correct_but_fact_missing_cases": int(len(group_missing)),
        "missing_fact_recovery_rate": float(
            (group_missing["hybrid_final_label"] == HYBRID_VERIFIED_BY_TEXT).mean()
            if len(group_missing) > 0 else 0.0
        )
    })

popularity_breakdown = pd.DataFrame(popularity_rows).sort_values(
    "kg_only_false_hallucination_rate",
    ascending=False
)

print("\nPopularity-level breakdown:")
display(popularity_breakdown)


# ============================================================
# 12. Clean important examples
# ============================================================

print("\nRecovered missing-fact examples:")
recovered_missing = missing_fact_df[
    missing_fact_df["hybrid_final_label"] == HYBRID_VERIFIED_BY_TEXT
].copy()

display(recovered_missing[[
    "question",
    "reference_answer",
    "model_raw_answer",
    "model_extracted_answer",
    "stage3_answer_case",
    "kg_only_prediction",
    "hybrid_final_label",
    "best_support_score",
    "best_contradiction_score",
    "best_evidence_strength",
    "best_answer_grounded_in_evidence",
    "best_evidence_text"
]].head(15))


print("\nCorrect missing-fact cases still not recovered:")
not_recovered_missing = missing_fact_df[
    missing_fact_df["hybrid_final_label"] != HYBRID_VERIFIED_BY_TEXT
].copy()

display(not_recovered_missing[[
    "question",
    "reference_answer",
    "model_raw_answer",
    "model_extracted_answer",
    "hybrid_final_label",
    "evidence_verdict",
    "best_support_score",
    "best_contradiction_score",
    "best_evidence_strength",
    "best_answer_grounded_in_evidence",
    "best_evidence_text"
]].head(15))


print("\nFalse acceptances preview:")
display(false_acceptances[[
    "question",
    "reference_answer",
    "model_raw_answer",
    "model_extracted_answer",
    "claim",
    "hybrid_final_label",
    "evidence_verdict",
    "best_support_score",
    "best_contradiction_score",
    "best_evidence_strength",
    "best_answer_grounded_in_evidence",
    "best_evidence_text"
]].head(15))


# ============================================================
# 13. Save outputs
# ============================================================

final_df.to_csv(STAGE5_FINAL_OUTPUT, index=False)

metrics = {
    "stage": "Stage 5 Final Safer Hybrid NLI Verification",
    "input_stage3_file": STAGE3_PATH,
    "input_stage4_file": STAGE4_PATH,
    "evidence_level_output_file": STAGE5_EVIDENCE_OUTPUT,
    "final_output_file": STAGE5_FINAL_OUTPUT,

    "num_total_rows": int(len(final_df)),
    "num_evaluated_rows": int(len(eval_df)),
    "num_valid_evidence_rows_for_nli": int(len(valid_evidence_df)),
    "num_aggregated_nli_rows": int(len(aggregated_nli_df)),

    "nli_model": NLI_MODEL_NAME,
    "thresholds": THRESHOLDS,

    "kg_only_accuracy": float(kg_only_accuracy),
    "hybrid_accuracy": float(hybrid_accuracy),

    "correct_model_answers": int(true_not_hallucination.sum()),
    "kg_only_false_hallucination_count": int(len(kg_false_hallucinations)),
    "hybrid_false_hallucination_count": int(len(hybrid_false_hallucinations)),
    "kg_only_false_hallucination_rate": float(kg_false_hallucination_rate),
    "hybrid_false_hallucination_rate": float(hybrid_false_hallucination_rate),

    "correct_but_fact_missing_cases": int(len(missing_fact_df)),
    "missing_fact_recovery_rate": float(missing_fact_recovery_rate),

    "false_acceptance_count": int(len(false_acceptances)),
    "false_acceptance_rate": float(false_acceptance_rate),

    "stage3_answer_case_distribution": final_df["stage3_answer_case"].value_counts().to_dict(),
    "kg_only_prediction_distribution": final_df["kg_only_prediction"].value_counts().to_dict(),
    "evidence_verdict_distribution": final_df["evidence_verdict"].value_counts().to_dict(),
    "hybrid_final_label_distribution": final_df["hybrid_final_label"].value_counts().to_dict(),

    "stage4_retrieval_scope_distribution": evidence_df["retrieval_scope"].value_counts().to_dict(),
    "stage4_evidence_strength_distribution": evidence_df["evidence_strength"].value_counts().to_dict(),

    "relation_breakdown": relation_breakdown.to_dict(orient="records"),
    "subject_popularity_breakdown": popularity_breakdown.to_dict(orient="records"),

    "kg_only_classification_report": classification_report(
        eval_df["reference_binary_label"],
        eval_df["kg_only_binary_prediction"],
        labels=labels,
        zero_division=0,
        output_dict=True
    ),

    "hybrid_classification_report": classification_report(
        eval_df["reference_binary_label"],
        eval_df["hybrid_binary_prediction"],
        labels=labels,
        zero_division=0,
        output_dict=True
    ),

    "kg_only_confusion_matrix_labels": labels,
    "kg_only_confusion_matrix": kg_cm.tolist(),

    "hybrid_confusion_matrix_labels": labels,
    "hybrid_confusion_matrix": hybrid_cm.tolist(),

    "label_definitions": {
        "hybrid_final_label": {
            HYBRID_SUPPORTED_BY_KG: "The answer is correct and the fact exists in the incomplete KG.",
            HYBRID_VERIFIED_BY_TEXT: "The KG did not support the answer, but Wikipedia evidence verified it.",
            HYBRID_CONTRADICTED_BY_TEXT: "Wikipedia evidence contradicts the model answer.",
            HYBRID_NOT_ENOUGH: "No strong enough evidence was found to support or contradict the answer.",
            UNUSABLE_MODEL_ANSWER: "The model output was not usable as a short entity answer."
        },
        "evidence_verdict": {
            EVIDENCE_SUPPORTED: "NLI found strong answer-grounded textual support.",
            EVIDENCE_CONTRADICTED: "NLI found strong relation-grounded textual contradiction.",
            EVIDENCE_NOT_ENOUGH: "Evidence was absent, weak, or not safely grounded."
        }
    },

    "important_safety_rules": {
        "relation_only_evidence_can_support": False,
        "relation_only_evidence_can_contradict": True,
        "support_requires_answer_grounding": True,
        "no_evidence_rows_stay_not_enough_evidence": True,
        "manual_alias_map_used": False,
        "uses_stage4_grounding_filters": True
    }
}

with open(STAGE5_METRICS_OUTPUT, "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved Stage 5 evidence-level NLI:", STAGE5_EVIDENCE_OUTPUT)
print("Saved Stage 5 final predictions:", STAGE5_FINAL_OUTPUT)
print("Saved Stage 5 metrics:", STAGE5_METRICS_OUTPUT)

print("\nStage 5 complete.")
print("\nFinal hybrid label distribution:")
print(final_df["hybrid_final_label"].value_counts())

print("\nKey final metrics:")
print(json.dumps({
    "kg_only_accuracy": metrics["kg_only_accuracy"],
    "hybrid_accuracy": metrics["hybrid_accuracy"],
    "kg_only_false_hallucination_rate": metrics["kg_only_false_hallucination_rate"],
    "hybrid_false_hallucination_rate": metrics["hybrid_false_hallucination_rate"],
    "missing_fact_recovery_rate": metrics["missing_fact_recovery_rate"],
    "false_acceptance_rate": metrics["false_acceptance_rate"]
}, indent=2))
# Final OUTPUT
# ============================================================
# FINAL RESULT SUMMARY TABLE
# Shows KG-only vs Hybrid counts and rates clearly
#
# Input:
# - data_final/stage5_hybrid_predictions.csv
#
# Output:
# - data_final/final_result_summary_table.csv
# - data_final/final_result_summary_metrics.json
# ============================================================

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

DATA_DIR = "data_final"

FINAL_PREDICTIONS_PATH = f"{DATA_DIR}/stage5_hybrid_predictions.csv"

SUMMARY_TABLE_PATH = f"{DATA_DIR}/final_result_summary_table.csv"
SUMMARY_JSON_PATH = f"{DATA_DIR}/final_result_summary_metrics.json"

assert os.path.exists(FINAL_PREDICTIONS_PATH), f"Missing file: {FINAL_PREDICTIONS_PATH}"

df = pd.read_csv(FINAL_PREDICTIONS_PATH)

print("Loaded final predictions:", len(df))
print("Columns:", list(df.columns))


# ============================================================
# 1. Keep only evaluated rows
# ============================================================

eval_df = df[
    df["stage3_answer_case"] != "UNUSABLE_MODEL_ANSWER"
].copy()

print("\nEvaluated rows:", len(eval_df))


# ============================================================
# 2. Build gold binary label
# ============================================================
# NOT_HALLUCINATION = model answer matches PopQA reference / aliases
# HALLUCINATION = model answer is wrong

eval_df["gold_binary_label"] = np.where(
    eval_df["model_answer_matches_reference"] == True,
    "NOT_HALLUCINATION",
    "HALLUCINATION"
)


# ============================================================
# 3. Build KG-only prediction
# ============================================================

eval_df["kg_only_binary_prediction"] = np.where(
    eval_df["kg_only_prediction"] == "SUPPORTED_BY_KG_ONLY",
    "NOT_HALLUCINATION",
    "HALLUCINATION"
)


# ============================================================
# 4. Build Hybrid prediction
# ============================================================

HYBRID_NOT_HALLUCINATION_LABELS = {
    "SUPPORTED_BY_KG",
    "VERIFIED_BY_TEXT_EVIDENCE"
}

eval_df["hybrid_binary_prediction"] = np.where(
    eval_df["hybrid_final_label"].isin(HYBRID_NOT_HALLUCINATION_LABELS),
    "NOT_HALLUCINATION",
    "HALLUCINATION"
)


# ============================================================
# 5. Main counts
# ============================================================

correct_model_answers = int(
    (eval_df["gold_binary_label"] == "NOT_HALLUCINATION").sum()
)

wrong_model_answers = int(
    (eval_df["gold_binary_label"] == "HALLUCINATION").sum()
)

missing_fact_cases = eval_df[
    eval_df["stage3_answer_case"] == "CORRECT_BUT_FACT_MISSING_FROM_KG"
].copy()

num_missing_fact_cases = int(len(missing_fact_cases))


# KG-only false hallucinations:
# model answer is correct, but KG-only says hallucination

kg_only_false_hallucinations = eval_df[
    (eval_df["gold_binary_label"] == "NOT_HALLUCINATION") &
    (eval_df["kg_only_binary_prediction"] == "HALLUCINATION")
].copy()

num_kg_only_false_hallucinations = int(len(kg_only_false_hallucinations))


# Hybrid false hallucinations:
# model answer is correct, but hybrid still says hallucination

hybrid_false_hallucinations = eval_df[
    (eval_df["gold_binary_label"] == "NOT_HALLUCINATION") &
    (eval_df["hybrid_binary_prediction"] == "HALLUCINATION")
].copy()

num_hybrid_false_hallucinations = int(len(hybrid_false_hallucinations))


# Missing facts recovered:
# among correct-but-missing facts, hybrid verifies them using text evidence

recovered_missing_facts = missing_fact_cases[
    missing_fact_cases["hybrid_final_label"] == "VERIFIED_BY_TEXT_EVIDENCE"
].copy()

num_recovered_missing_facts = int(len(recovered_missing_facts))


# Missing facts still missed

still_missed_missing_facts = missing_fact_cases[
    missing_fact_cases["hybrid_final_label"] != "VERIFIED_BY_TEXT_EVIDENCE"
].copy()

num_still_missed_missing_facts = int(len(still_missed_missing_facts))


# False acceptances:
# model answer is wrong, but hybrid accepts it as not hallucination

false_acceptances = eval_df[
    (eval_df["gold_binary_label"] == "HALLUCINATION") &
    (eval_df["hybrid_binary_prediction"] == "NOT_HALLUCINATION")
].copy()

num_false_acceptances = int(len(false_acceptances))


# ============================================================
# 6. Rates
# ============================================================

kg_only_accuracy = accuracy_score(
    eval_df["gold_binary_label"],
    eval_df["kg_only_binary_prediction"]
)

hybrid_accuracy = accuracy_score(
    eval_df["gold_binary_label"],
    eval_df["hybrid_binary_prediction"]
)

kg_only_false_hallucination_rate = (
    num_kg_only_false_hallucinations / correct_model_answers
    if correct_model_answers > 0 else 0.0
)

hybrid_false_hallucination_rate = (
    num_hybrid_false_hallucinations / correct_model_answers
    if correct_model_answers > 0 else 0.0
)

missing_fact_recovery_rate = (
    num_recovered_missing_facts / num_missing_fact_cases
    if num_missing_fact_cases > 0 else 0.0
)

false_acceptance_rate = (
    num_false_acceptances / wrong_model_answers
    if wrong_model_answers > 0 else 0.0
)


# ============================================================
# 7. Final comparison table
# ============================================================

summary_table = pd.DataFrame([
    {
        "method": "KG-only baseline",
        "accuracy": kg_only_accuracy,
        "accuracy_percent": round(kg_only_accuracy * 100, 2),
        "false_hallucinations": num_kg_only_false_hallucinations,
        "false_hallucination_denominator": correct_model_answers,
        "false_hallucination_rate": kg_only_false_hallucination_rate,
        "false_hallucination_rate_percent": round(kg_only_false_hallucination_rate * 100, 2),
        "missing_facts_recovered": 0,
        "missing_fact_total": num_missing_fact_cases,
        "missing_fact_recovery_rate": 0.0,
        "missing_fact_recovery_rate_percent": 0.0,
        "false_acceptances": 0,
        "false_acceptance_denominator": wrong_model_answers,
        "false_acceptance_rate": 0.0,
        "false_acceptance_rate_percent": 0.0
    },
    {
        "method": "Hybrid KG + text evidence + NLI",
        "accuracy": hybrid_accuracy,
        "accuracy_percent": round(hybrid_accuracy * 100, 2),
        "false_hallucinations": num_hybrid_false_hallucinations,
        "false_hallucination_denominator": correct_model_answers,
        "false_hallucination_rate": hybrid_false_hallucination_rate,
        "false_hallucination_rate_percent": round(hybrid_false_hallucination_rate * 100, 2),
        "missing_facts_recovered": num_recovered_missing_facts,
        "missing_fact_total": num_missing_fact_cases,
        "missing_fact_recovery_rate": missing_fact_recovery_rate,
        "missing_fact_recovery_rate_percent": round(missing_fact_recovery_rate * 100, 2),
        "false_acceptances": num_false_acceptances,
        "false_acceptance_denominator": wrong_model_answers,
        "false_acceptance_rate": false_acceptance_rate,
        "false_acceptance_rate_percent": round(false_acceptance_rate * 100, 2)
    }
])

print("\nFinal result summary table:")
display(summary_table)


# ============================================================
# 8. Reviewer-friendly compact table
# ============================================================

compact_table = pd.DataFrame([
    {
        "Method": "KG-only baseline",
        "Accuracy": f"{kg_only_accuracy * 100:.2f}%",
        "False hallucinations": f"{num_kg_only_false_hallucinations} / {correct_model_answers}",
        "False hallucination rate": f"{kg_only_false_hallucination_rate * 100:.2f}%",
        "Missing facts recovered": f"0 / {num_missing_fact_cases}",
        "Recovery rate": "0.00%",
        "False acceptances": "0"
    },
    {
        "Method": "Hybrid KG + text evidence + NLI",
        "Accuracy": f"{hybrid_accuracy * 100:.2f}%",
        "False hallucinations": f"{num_hybrid_false_hallucinations} / {correct_model_answers}",
        "False hallucination rate": f"{hybrid_false_hallucination_rate * 100:.2f}%",
        "Missing facts recovered": f"{num_recovered_missing_facts} / {num_missing_fact_cases}",
        "Recovery rate": f"{missing_fact_recovery_rate * 100:.2f}%",
        "False acceptances": f"{num_false_acceptances} / {wrong_model_answers}"
    }
])

print("\nCompact table for paper/report:")
display(compact_table)


# ============================================================
# 9. Clear result sentence
# ============================================================

result_sentence = (
    f"The KG-only baseline falsely flagged {num_kg_only_false_hallucinations} "
    f"correct model answers as hallucinations because their facts were missing "
    f"from the incomplete KG. The hybrid method successfully recovered "
    f"{num_recovered_missing_facts} of these {num_missing_fact_cases} missing-fact cases, "
    f"reducing false hallucinations from {num_kg_only_false_hallucinations} to "
    f"{num_hybrid_false_hallucinations}, with a missing-fact recovery rate of "
    f"{missing_fact_recovery_rate * 100:.2f}% and only {num_false_acceptances} false acceptance."
)

print("\nReviewer-friendly result sentence:")
print(result_sentence)


# ============================================================
# 10. Save outputs
# ============================================================

summary_table.to_csv(SUMMARY_TABLE_PATH, index=False)

summary_metrics = {
    "num_evaluated_rows": int(len(eval_df)),
    "correct_model_answers": correct_model_answers,
    "wrong_model_answers": wrong_model_answers,
    "missing_fact_cases": num_missing_fact_cases,
    "kg_only_accuracy": float(kg_only_accuracy),
    "hybrid_accuracy": float(hybrid_accuracy),
    "kg_only_false_hallucinations": num_kg_only_false_hallucinations,
    "hybrid_false_hallucinations": num_hybrid_false_hallucinations,
    "kg_only_false_hallucination_rate": float(kg_only_false_hallucination_rate),
    "hybrid_false_hallucination_rate": float(hybrid_false_hallucination_rate),
    "missing_facts_recovered": num_recovered_missing_facts,
    "missing_facts_still_missed": num_still_missed_missing_facts,
    "missing_fact_recovery_rate": float(missing_fact_recovery_rate),
    "false_acceptances": num_false_acceptances,
    "false_acceptance_rate": float(false_acceptance_rate),
    "result_sentence": result_sentence
}

with open(SUMMARY_JSON_PATH, "w") as f:
    json.dump(summary_metrics, f, indent=2)

print("\nSaved summary table:", SUMMARY_TABLE_PATH)
print("Saved summary metrics:", SUMMARY_JSON_PATH)


# ============================================================
# 11. Optional: save example files for paper discussion
# ============================================================

recovered_missing_facts.to_csv(
    f"{DATA_DIR}/final_recovered_missing_fact_examples.csv",
    index=False
)

still_missed_missing_facts.to_csv(
    f"{DATA_DIR}/final_still_missed_missing_fact_examples.csv",
    index=False
)

false_acceptances.to_csv(
    f"{DATA_DIR}/final_false_acceptance_examples.csv",
    index=False
)

print("\nSaved example files:")
print("-", f"{DATA_DIR}/final_recovered_missing_fact_examples.csv")
print("-", f"{DATA_DIR}/final_still_missed_missing_fact_examples.csv")
print("-", f"{DATA_DIR}/final_false_acceptance_examples.csv")
# THE END
