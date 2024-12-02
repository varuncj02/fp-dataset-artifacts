import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
import evaluate
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json
import pandas as pd

NUM_PREPROCESSING_WORKERS = 2


def load_and_preprocess_contrast_dataset(file_path, tokenizer, max_length):
    """
    Load and preprocess the contrast dataset.
    """
    import pandas as pd
    contrast_data = pd.read_csv(file_path)
    dataset = datasets.Dataset.from_dict({
        "premise": contrast_data["premise"].tolist(),
        "hypothesis": contrast_data["contrast_hypothesis"].tolist(),
        "label": [int(label) for label in contrast_data["contrast_label"]]
    })
    processed_dataset = dataset.map(
        lambda exs: prepare_dataset_nli(exs, tokenizer, max_length),
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=dataset.column_names
    )
    return processed_dataset


def fine_tune_on_contrast(trainer, contrast_dataset):
    """
    Fine-tune the model on the contrast dataset.
    """
    print("Fine-tuning on contrast dataset...")
    trainer.train_dataset = contrast_dataset
    trainer.train()
    trainer.save_model()

def main():
    argp = HfArgumentParser(TrainingArguments)
    # Existing arguments
    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to evaluate.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--task', type=str, choices=['nli', 'qa'], required=True,
                      help="""This argument specifies which task to evaluate on.
        Pass "nli" for natural language inference or "qa" for question answering.""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--contrast_dataset', type=str, default=None,
                      help="""Path to the contrast dataset CSV file for evaluation.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during evaluation.""")
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')

    training_args, args = argp.parse_args_into_dataclasses()

    # Dataset selection for the original dataset
    dataset = None
    if args.dataset and (args.dataset.endswith('.json') or args.dataset.endswith('.jsonl')):
        dataset = datasets.load_dataset('json', data_files=args.dataset)
    elif args.dataset:
        default_datasets = {'qa': ('squad',), 'nli': ('snli',)}
        dataset_id = tuple(args.dataset.split(':')) if args.dataset else default_datasets[args.task]
        dataset = datasets.load_dataset(*dataset_id)

    # Load and preprocess contrast dataset if provided
    contrast_dataset = None
    if args.contrast_dataset:
        import pandas as pd
        contrast_data = pd.read_csv(args.contrast_dataset)
        contrast_dataset = datasets.Dataset.from_dict({
            "premise": contrast_data["premise"].tolist(),
            "hypothesis": contrast_data["contrast_hypothesis"].tolist(),
            "label": [int(label) for label in contrast_data["contrast_label"]]
        })

    # Preprocessing functions
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    prepare_eval_dataset = None
    if args.task == 'qa':
        prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    elif args.task == 'nli':
        prepare_eval_dataset = lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)

    # Preprocess datasets for evaluation
    eval_dataset_featurized = None
    contrast_dataset_featurized = None
    if dataset:
        eval_dataset_featurized = dataset["validation"].map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=dataset["validation"].column_names
        )
    if contrast_dataset:
        contrast_dataset_featurized = contrast_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=contrast_dataset.column_names
        )

    # Initialize model
    model_classes = {'qa': AutoModelForQuestionAnswering, 'nli': AutoModelForSequenceClassification}
    model_class = model_classes[args.task]
    model = model_class.from_pretrained(args.model)

    # Initialize Trainer
    compute_metrics = None
    if args.task == 'qa':
        metric = evaluate.load('squad')
        compute_metrics = lambda eval_preds: metric.compute(predictions=eval_preds.predictions, references=eval_preds.label_ids)
    elif args.task == 'nli':
        compute_metrics = compute_accuracy

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Evaluate on original validation dataset
    if training_args.do_eval and eval_dataset_featurized:
        print("Evaluating on original validation dataset...")
        results = trainer.evaluate(eval_dataset=eval_dataset_featurized)
        print("Evaluation results on original validation dataset:", results)

    # Evaluate on contrast dataset
    if training_args.do_eval and contrast_dataset_featurized:
        print("Evaluating on contrast dataset...")
        contrast_results = trainer.evaluate(eval_dataset=contrast_dataset_featurized)
        print("Evaluation results on contrast dataset:", contrast_results)

        # Save results to the output directory
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(os.path.join(training_args.output_dir, "contrast_eval_metrics.json"), "w") as f:
            json.dump(contrast_results, f, indent=4)


if __name__ == '__main__':
    main()