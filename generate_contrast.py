import pandas as pd
from datasets import load_dataset

def create_contrast_examples(dataset, num_samples=100):
    contrast_examples = []
    for i, example in enumerate(dataset["train"]):
        if i >= num_samples:
            break
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        label = example["label"]

        if not hypothesis:
            continue

        contrast_hypothesis = hypothesis
        if label == 0:  # Entailment
            contrast_hypothesis = hypothesis.replace("is", "is not")
            new_label = 2
        elif label == 2:  # Contradiction
            contrast_hypothesis = hypothesis.replace("is not", "is")
            new_label = 0
        else:
            contrast_hypothesis = hypothesis + " perhaps."
            new_label = 1

        contrast_examples.append({
            "premise": premise,
            "original_hypothesis": hypothesis,
            "contrast_hypothesis": contrast_hypothesis,
            "original_label": label,
            "contrast_label": new_label
        })
    return contrast_examples

# Load SNLI and generate examples
snli = load_dataset("snli")
contrast_examples = create_contrast_examples(snli, num_samples=500)

# Save to CSV
contrast_df = pd.DataFrame(contrast_examples)
contrast_df.to_csv("contrast_examples.csv", index=False)
print("Contrast examples saved as contrast_examples.csv")
