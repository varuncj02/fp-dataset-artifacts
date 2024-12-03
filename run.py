import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
import evaluate
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy, \
    analyze_errors
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from helpers import analyze_mistakes, analyze_misclassification_types

NUM_PREPROCESSING_WORKERS = 2

def load_and_preprocess_contrast_dataset(file_path, tokenizer, max_length):
    """
    Load and preprocess the contrast dataset for NLI evaluation.
    """
    contrast_data = pd.read_csv(file_path, usecols=["premise", "hypothesis", "label"])

    # Convert to HuggingFace Dataset
    dataset = datasets.Dataset.from_pandas(contrast_data)

    # Map data for Electra
    processed_dataset = dataset.map(
        lambda exs: prepare_dataset_nli(exs, tokenizer, max_length),
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=dataset.column_names
    )
    return dataset, processed_dataset


def analyze_results(contrast_dataset_path, predictions_path, output_dir):
    """
    Enhanced analysis of the baseline Electra model's performance on the contrast dataset.

    Args:
    - contrast_dataset_path (str): Path to the contrast dataset (CSV).
    - predictions_path (str): Path to the predictions JSON file.
    - output_dir (str): Directory to save analysis results and visuals.
    """
    # Load contrast dataset and predictions
    contrast_data = pd.read_csv(contrast_dataset_path, usecols=["premise", "hypothesis", "label"])
    with open(predictions_path, "r") as f:
        predictions = json.load(f)

    # Add predictions to dataset
    contrast_data["predicted_label"] = predictions["predicted_label"]
    contrast_data["is_correct"] = contrast_data["label"] == contrast_data["predicted_label"]

    # Analysis Categories
    def contains_negation(text):
        negations = ["not", "never", "no", "nobody", "nothing", "nowhere", "neither"]
        return any(neg in text.lower() for neg in negations)

    def is_long_hypothesis(hypothesis):
        return len(hypothesis.split()) > 15

    def unrelatedness_score(premise, hypothesis):
        p_tokens = set(premise.lower().split())
        h_tokens = set(hypothesis.lower().split())
        return 1 - (len(p_tokens & h_tokens) / max(len(p_tokens | h_tokens), 1))

    # Compute analysis metrics
    contrast_data["contains_negation"] = contrast_data["hypothesis"].apply(contains_negation)
    contrast_data["is_long_hypothesis"] = contrast_data["hypothesis"].apply(is_long_hypothesis)
    contrast_data["word_overlap"] = contrast_data.apply(
        lambda row: len(set(row["premise"].split()) & set(row["hypothesis"].split())), axis=1
    )
    contrast_data["unrelatedness_score"] = contrast_data.apply(
        lambda row: unrelatedness_score(row["premise"], row["hypothesis"]), axis=1
    )
    contrast_data["is_unrelated"] = contrast_data["unrelatedness_score"] > 0.8

    # Compute metrics for each category
    categories = {
        "Negation": contrast_data[contrast_data["contains_negation"]],
        "Long Hypothesis": contrast_data[contrast_data["is_long_hypothesis"]],
        "Unrelated Hypothesis": contrast_data[contrast_data["is_unrelated"]],
        "Word Overlap <= 3": contrast_data[contrast_data["word_overlap"] <= 3],
    }
    category_stats = {}
    detailed_examples = {}

    for category, df in categories.items():
        category_stats[category] = {
            "accuracy": float(df["is_correct"].mean()),
            "precision": float((df["is_correct"] & (df["predicted_label"] == df["label"])).sum() / max(df["predicted_label"].sum(), 1)),
            "recall": float((df["is_correct"] & (df["predicted_label"] == df["label"])).sum() / max(df["label"].sum(), 1)),
            "f1": float(2 * ((df["is_correct"].mean() * (df["is_correct"] & (df["predicted_label"] == df["label"])).sum()) / max(df["predicted_label"].sum() + df["label"].sum(), 1))),
            "error_count": int((~df["is_correct"]).sum()),
            "example_count": int(len(df))
        }
        # Save specific examples
        detailed_examples[category] = {
            "correct_examples": df[df["is_correct"]].head(3).to_dict(orient="records"),
            "incorrect_examples": df[~df["is_correct"]].head(3).to_dict(orient="records"),
        }

    # Save category statistics
    with open(f"{output_dir}/category_stats.json", "w") as f:
        json.dump(category_stats, f, indent=4)

    # Save detailed examples
    with open(f"{output_dir}/detailed_examples.json", "w") as f:
        json.dump(detailed_examples, f, indent=4)

    # Visualizations
    # 1. Bar chart for accuracy
    plt.figure(figsize=(8, 6))
    plt.bar(category_stats.keys(), [stat["accuracy"] for stat in category_stats.values()])
    plt.title("Accuracy by Category")
    plt.ylabel("Accuracy")
    plt.savefig(f"{output_dir}/accuracy_by_category.png")
    plt.close()

    # 2. Histogram of error counts
    plt.figure(figsize=(8, 6))
    plt.bar(category_stats.keys(), [stat["error_count"] for stat in category_stats.values()])
    plt.title("Error Counts by Category")
    plt.ylabel("Error Count")
    plt.savefig(f"{output_dir}/error_counts_by_category.png")
    plt.close()

    # 3. Word overlap vs. accuracy
    plt.figure(figsize=(8, 6))
    plt.scatter(contrast_data["word_overlap"], contrast_data["is_correct"], alpha=0.5)
    plt.title("Word Overlap vs. Accuracy")
    plt.xlabel("Word Overlap")
    plt.ylabel("Correct Prediction (1=True, 0=False)")
    plt.savefig(f"{output_dir}/word_overlap_vs_accuracy.png")
    plt.close()


def main():
    argp = HfArgumentParser(TrainingArguments)

    # Existing arguments
    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to evaluate.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--task', type=str, choices=['nli'], required=True,
                      help="""This argument specifies which task to evaluate on.
        Pass "nli" for natural language inference.""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during evaluation.""")
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')

    training_args, args = argp.parse_args_into_dataclasses()

    # Preprocessing functions
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Select the dataset
    if args.dataset:
        if args.dataset.endswith('.csv'):
            # Load and preprocess the contrast dataset
            print(f"Loading contrast dataset from {args.dataset}...")
            contrast_dataset, eval_dataset_featurized = load_and_preprocess_contrast_dataset(
                args.dataset, tokenizer, args.max_length
            )
        else:
            # Default behavior for standard datasets
            default_datasets = {'qa': ('squad',), 'nli': ('snli',)}
            dataset_id = tuple(args.dataset.split(':')) if args.dataset else default_datasets[args.task]
            eval_split = 'validation_matched' if dataset_id == ('glue', 'mnli') else 'validation'

            print(f"Loading default dataset: {dataset_id}")
            dataset = datasets.load_dataset(*dataset_id)
            eval_dataset = dataset[eval_split]
            eval_dataset_featurized = eval_dataset.map(
                lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length),
                batched=True,
                num_proc=NUM_PREPROCESSING_WORKERS,
                remove_columns=eval_dataset.column_names
            )
    else:
        raise ValueError("You must provide a dataset using the --dataset argument.")

    # Initialize model
    model_classes = {'nli': AutoModelForSequenceClassification}
    model_class = model_classes[args.task]
    model = model_class.from_pretrained(args.model)

    # Initialize Trainer
    compute_metrics = compute_accuracy
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Evaluate
    if training_args.do_eval:
        print("Evaluating model...")
        results = trainer.evaluate(eval_dataset=eval_dataset_featurized)
        print("Evaluation Results:", results)

        # Predict explicitly to save predictions
        predictions = trainer.predict(eval_dataset_featurized)
        predicted_labels = predictions.predictions.argmax(axis=1)

        # Save evaluation results
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(os.path.join(training_args.output_dir, "evaluation_results.json"), "w") as f:
            json.dump(results, f, indent=4)

        # Save predictions
        prediction_output = {
            "predicted_label": predicted_labels.tolist(),
            "label_ids": predictions.label_ids.tolist(),
        }
        with open(os.path.join(training_args.output_dir, "predictions.json"), "w") as f:
            json.dump(prediction_output, f, indent=4)

        # Perform analysis
        print("Analyzing evaluation results...")
        analyze_results(
            args.dataset,
            os.path.join(training_args.output_dir, "predictions.json"),
            training_args.output_dir
        )


if __name__ == "__main__":
    main()