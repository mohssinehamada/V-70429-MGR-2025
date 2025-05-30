import os
import json
import torch
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "fake_news_model/targeted_results"
MODEL_NAME = "roberta-base" 

def extract_label_from_response(response):
    """
    extract label with improved robustness, specifically designed to handle known formats
    in the dataset including the "Verdict:" format
    """
    if not response:
        return "UNKNOWN"
    
    response_lower = response.lower()
    
   
    if "verdict:" in response_lower:
        verdict_parts = response_lower.split("verdict:")
        if len(verdict_parts) > 1:
            verdict = verdict_parts[1].strip()
            
           
            if any(phrase in verdict for phrase in [" true", "true.", "true "]) and not any(phrase in verdict for phrase in ["partially", "half", "mostly"]):
                return "TRUE"
            elif any(phrase in verdict for phrase in [" false", "false.", "false "]):
                return "FALSE"
            elif any(phrase in verdict for phrase in ["partially true", "half true", "partly true", "half-true"]):
                return "PARTIALLY TRUE"
    
   
    if "true" in response_lower and not any(phrase in response_lower for phrase in ["partially true", "half true", "partly true", "half-true"]):
        return "TRUE"
    elif "false" in response_lower:
        return "FALSE"
    elif any(phrase in response_lower for phrase in ["partially true", "half true", "partly true", "half-true"]):
        return "PARTIALLY TRUE"
    
   
    return "UNKNOWN"

def load_and_preprocess_data(json_file):
    
    logger.info(f"Loading data from {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
   
    labeled_data = []
    label_mapping_examples = {}
    
   
    for item in data:
        instruction = item.get('instruction', '')
        response = item.get('response', '')
        
       
        label = extract_label_from_response(response)
        
       
        if label != "UNKNOWN" and label not in label_mapping_examples:
            label_mapping_examples[label] = response[:100] + "..."
        
       
        if label != "UNKNOWN":
            labeled_data.append({
                "text": instruction,
                "label": label,
                "original_response": response[:50] + "..." 
            })
    
   
    logger.info("Label mapping examples:")
    for label, example in label_mapping_examples.items():
        logger.info(f"{label}: {example}")
    
    
    label_counts = {}
    for item in labeled_data:
        label = item["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    logger.info(f"Label distribution: {label_counts}")
    
    
    train_data, val_data = train_test_split(
        labeled_data, 
        test_size=0.2, 
        stratify=[item["label"] for item in labeled_data],
        random_state=42
    )
    
    logger.info(f"Train set: {len(train_data)} examples")
    logger.info(f"Validation set: {len(val_data)} examples")
    
    
    train_dataset = Dataset.from_dict({
        "text": [item["text"] for item in train_data],
        "label": [item["label"] for item in train_data]
    })
    
    val_dataset = Dataset.from_dict({
        "text": [item["text"] for item in val_data],
        "label": [item["label"] for item in val_data]
    })
    
    return train_dataset, val_dataset, label_counts

def tokenize_and_prepare(dataset, tokenizer, max_length=512):
    
   
    label_names = sorted(set(dataset["label"]))
    label_map = {label: i for i, label in enumerate(label_names)}
    
    logger.info(f"Label mapping: {label_map}")
    

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        tokenized["labels"] = [label_map[label] for label in examples["label"]]
        return tokenized
    
  
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "label"]
    )
    
    return tokenized_dataset, label_map, label_names

def compute_metrics(eval_pred):
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
  
    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    accuracy = accuracy_score(labels, predictions)
    
    
    metrics = {
        'accuracy': float(accuracy),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
    }
    
    
    for i in range(len(per_class_precision)):
        metrics[f'class_{i}_precision'] = float(per_class_precision[i])
        metrics[f'class_{i}_recall'] = float(per_class_recall[i])
        metrics[f'class_{i}_f1'] = float(per_class_f1[i])
        metrics[f'class_{i}_support'] = int(support[i])
    
    
    cm = confusion_matrix(labels, predictions)
    logger.info(f"Confusion matrix:\n{cm}")
    
    return metrics

def visualize_results(trainer_history, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    train_loss = [x['loss'] for x in trainer_history if 'loss' in x]
    steps = [x['step'] for x in trainer_history if 'loss' in x]
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_loss)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()
    
   
    eval_metrics = {
        'accuracy': [],
        'f1': [],
        'precision': [],
        'recall': []
    }
    eval_steps = []
    
    
    for log in trainer_history:
        if 'eval_accuracy' in log:
            eval_steps.append(log['step'])
            eval_metrics['accuracy'].append(log['eval_accuracy'])
            eval_metrics['f1'].append(log['eval_f1'])
            eval_metrics['precision'].append(log['eval_precision'])
            eval_metrics['recall'].append(log['eval_recall'])
    
    if eval_steps:  
        plt.figure(figsize=(12, 8))
       
        for metric_name, values in eval_metrics.items():
            plt.plot(eval_steps, values, label=metric_name.capitalize(), linewidth=2)
        
        plt.xlabel('Training Steps')
        plt.ylabel('Score')
        plt.title('Model Evaluation Metrics')
        plt.legend()
        plt.grid(True)
       
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        plt.axhline(y=0.75, color='gray', linestyle='--', alpha=0.3)
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
        
      
        plt.ylim(0, 1.1)
        
 
        plt.savefig(os.path.join(output_dir, 'eval_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        metrics_df = pd.DataFrame({
            'step': eval_steps,
            **eval_metrics
        })
        metrics_df.to_csv(os.path.join(output_dir, 'metrics_history.csv'), index=False)

def train_and_evaluate():
    
 
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(OUTPUT_DIR, "training_log.txt"))
    logger.addHandler(file_handler)
    
    logger.info("Starting targeted training for fake news detection")
    
 
    train_dataset, val_dataset, label_counts = load_and_preprocess_data(
        "fake_news_model/data/processed/train_formatted.json"
    )
    

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    

    tokenized_train, label_map, label_names = tokenize_and_prepare(train_dataset, tokenizer)
    tokenized_val, _, _ = tokenize_and_prepare(val_dataset, tokenizer)
    
 
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump({"label_map": label_map, "label_names": label_names}, f, indent=2)
    
  
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    logger.info(f"Using device: {device}")
    
    
    num_labels = len(label_map)
    class_counts = [label_counts.get(label, 0) for label in label_names]
    total_samples = sum(class_counts)
    class_weights = [total_samples / (num_labels * count) if count > 0 else 1.0 for count in class_counts]
    
   
    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    logger.info(f"Class weights: {class_weights}")
    
   
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    )
    
   
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=50,              
        evaluation_strategy="steps",   
        eval_steps=100,               
        save_strategy="steps",        
        save_steps=100,               
        save_total_limit=2,           
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none"              
    )
    
    
    class WeightedLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(**inputs)
            logits = outputs.get("logits")
            labels = inputs.get("labels")
       
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weight_tensor)
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            
            return (loss, outputs) if return_outputs else loss
    
   
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
 
    logger.info("Starting training")
    train_result = trainer.train()
    
    
    logger.info("Creating visualizations")
    visualize_results(trainer.state.log_history, OUTPUT_DIR)

    logger.info("Performing final evaluation")
    metrics = trainer.evaluate()
    
   
    logger.info(f"Final evaluation metrics: {metrics}")
 
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "tokenizer"))
    
    with open(os.path.join(OUTPUT_DIR, "final_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    
    with open(os.path.join(OUTPUT_DIR, "training_summary.txt"), "w") as f:
        f.write("TARGETED FAKE NEWS DETECTION - TRAINING SUMMARY\n")
        f.write("============================================\n\n")
        
        f.write("DATA INFORMATION:\n")
        f.write(f"- Label distribution: {label_counts}\n")
        f.write(f"- Training examples: {len(tokenized_train)}\n")
        f.write(f"- Validation examples: {len(tokenized_val)}\n\n")
        
        f.write("MODEL INFORMATION:\n")
        f.write(f"- Base model: {MODEL_NAME}\n")
        f.write(f"- Number of labels: {num_labels}\n")
        f.write(f"- Label mapping: {label_map}\n\n")
        
        f.write("TRAINING PARAMETERS:\n")
        f.write(f"- Learning rate: {training_args.learning_rate}\n")
        f.write(f"- Batch size: {training_args.per_device_train_batch_size}\n")
        f.write(f"- Epochs: {training_args.num_train_epochs}\n")
        f.write(f"- Weight decay: {training_args.weight_decay}\n")
        f.write(f"- Class weights: {class_weights}\n\n")
        
        f.write("RESULTS:\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"- {metric_name}: {metric_value:.4f}\n")
        
    logger.info(f"Training complete. Results saved to {OUTPUT_DIR}")
    return metrics

if __name__ == "__main__":
    train_and_evaluate() 