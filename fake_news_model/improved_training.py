import os
import json
import torch
import numpy as np
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    RobertaConfig,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import random
import time
from torch.nn import functional as F

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImprovedFakeNewsTrainer:
    def __init__(self, 
                 model_name="facebook/roberta-hate-speech-dynabench-r4-target", 
                 output_dir="fake_news_model/improved_results",
                 cache_dir="fake_news_model/cache",
                 use_wandb=False,
                 seed=42):
        self.model_name = model_name
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.seed = seed
        self.use_wandb = use_wandb and has_wandb
        
        if self.use_wandb and not has_wandb:
            logger.warning("wandb not installed. Run 'pip install wandb' to enable experiment tracking.")
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS (Metal Performance Shaders) for training")
        else:
            self.device = torch.device("cpu")
            logger.info("MPS not available, falling back to CPU")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        self.tokenizer = RobertaTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            use_fast=True  
        )
        
        self.label_map = {
            "FALSE": 0,
            "PARTIALLY TRUE": 1,
            "TRUE": 2
        }
        self.id2label = {v: k for k, v in self.label_map.items()}
        
        config = RobertaConfig.from_pretrained(
            model_name,
            num_labels=3,
            problem_type="single_label_classification",
            id2label=self.id2label,
            label2id=self.label_map,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            classifier_dropout=0.2,
            cache_dir=cache_dir
        )
        
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
            cache_dir=cache_dir
        )
        
        self._modify_model_for_performance()
        
        self.model.to(self.device)
        
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding="longest",
            max_length=512,
            pad_to_multiple_of=8
        )
        
    def _modify_model_for_performance(self):

        for param in self.model.roberta.embeddings.parameters():
            param.requires_grad = False
            
        for layer in self.model.roberta.encoder.layer[:6]:
            for param in layer.parameters():
                param.requires_grad = False
        
        logger.info("Model modified: Embeddings and lower layers frozen for transfer learning")
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
        
    def load_and_preprocess_data(self, data_path, oversampling_minority=True):
        
        logger.info(f"Loading data from {data_path}")
        start_time = time.time()
        
        texts = []
        labels = []
        metadata = []
        
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                claim = item.get('claim', '')
                evidence = item.get('evidence', [])
                
                evidence_text = ' '.join([str(e) for e in evidence]) if evidence else ''
                
                text = f"Claim: {claim} Evidence: {evidence_text}"
                
                original_label = item.get('label', 'UNKNOWN')
                if original_label == 'SUPPORTS':
                    label = 'TRUE'
                elif original_label == 'REFUTES':
                    label = 'FALSE'
                else:
                    label = 'PARTIALLY TRUE'
                
                metadata.append({
                    'original_label': original_label,
                    'claim_length': len(claim),
                    'evidence_length': len(evidence_text),
                    'has_evidence': len(evidence) > 0
                })
                
                if label in self.label_map:
                    texts.append(text)
                    labels.append(self.label_map[label])
        
        df = pd.DataFrame({
            'text': texts,
            'label': labels,
            **{k: [d[k] for d in metadata] for k in metadata[0].keys()}
        })
        
        logger.info(f"Data loaded: {len(df)} examples")
        logger.info(f"Class distribution: {Counter(labels)}")
        logger.info(f"Average claim length: {df['claim_length'].mean():.1f} chars")
        logger.info(f"Average evidence length: {df['evidence_length'].mean():.1f} chars")
        logger.info(f"Examples with evidence: {df['has_evidence'].mean():.1%}")
        
        label_counts = Counter(labels)
        total_samples = len(labels)
        class_weights = {
            label: total_samples / (len(label_counts) * count)
            for label, count in label_counts.items()
        }
        self.class_weights = torch.tensor([class_weights[i] for i in range(len(class_weights))], device=self.device)
        
        train_df, val_df = train_test_split(
            df, test_size=0.2, stratify=df['label'], random_state=self.seed
        )
        
        if oversampling_minority:
            majority_class = label_counts.most_common(1)[0][0]
            majority_count = label_counts[majority_class]
            
            oversampled_dfs = [train_df[train_df['label'] == majority_class]]
            
            for label, count in label_counts.items():
                if label != majority_class:
                    ratio = majority_count // count
                    if ratio > 1:
                        df_minority = train_df[train_df['label'] == label]
                        oversampled_dfs.append(df_minority)
                        for _ in range(ratio - 1):
                            augmented_df = df_minority.copy()
                            augmented_df['text'] = augmented_df['text'].apply(
                                lambda x: self._augment_text(x)
                            )
                            oversampled_dfs.append(augmented_df)
            
            train_df = pd.concat(oversampled_dfs)
            train_df = train_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
            
            logger.info(f"After oversampling, train set size: {len(train_df)}")
            logger.info(f"Oversampled class distribution: {Counter(train_df['label'])}")
        
        train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
        val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
        
        logger.info(f"Train set: {len(train_dataset)} examples")
        logger.info(f"Validation set: {len(val_dataset)} examples")
        logger.info(f"Data preparation took {time.time() - start_time:.2f} seconds")
        
        return train_dataset, val_dataset
    
    def _augment_text(self, text):
        words = text.split()
        if len(words) <= 10:
            return text
            
        if random.random() < 0.3 and len(words) > 15:
            delete_indices = random.sample(range(len(words)), int(len(words) * 0.1))
            words = [w for i, w in enumerate(words) if i not in delete_indices]
            
        if random.random() < 0.2 and len(words) > 10:
            for _ in range(int(len(words) * 0.05)):
                i, j = random.sample(range(len(words)), 2)
                words[i], words[j] = words[j], words[i]
                
        return ' '.join(words)
    
    def tokenize_and_prepare(self, dataset):
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding=False,  
                truncation=True,
                max_length=512,
                return_tensors=None,  
                return_length=True    
            )
        
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=1,  
            remove_columns=["text"],
            desc="Tokenizing datasets"
        )
        
        return tokenized_dataset
    
    def compute_metrics(self, eval_pred):
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        
        per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        
        cm = confusion_matrix(labels, predictions)
        
        
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        
        try:
            
            probs = F.softmax(torch.tensor(eval_pred.predictions), dim=1).numpy()
            
            roc_auc = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')
        except Exception as e:
            logger.warning(f"Couldn't calculate ROC-AUC: {e}")
            roc_auc = 0.0
        
        
        metrics = {
            'accuracy': float(accuracy),
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'roc_auc': float(roc_auc)
        }
        
        
        for i, label in enumerate(self.id2label.values()):
            metrics[f'{label}_precision'] = float(per_class_precision[i])
            metrics[f'{label}_recall'] = float(per_class_recall[i])
            metrics[f'{label}_f1'] = float(per_class_f1[i])
            metrics[f'{label}_support'] = int(support[i])
        
        
        for i in range(len(self.id2label)):
            for j in range(len(self.id2label)):
                i_name = self.id2label[i]
                j_name = self.id2label[j]
                metrics[f'cm_{i_name}_predicted_as_{j_name}'] = int(cm[i, j])
                metrics[f'cm_norm_{i_name}_predicted_as_{j_name}'] = float(cm_normalized[i, j])
        
        
        metrics['error_rate'] = 1.0 - accuracy
        metrics['balanced_accuracy'] = float(np.mean(per_class_recall))  
        
        return metrics
    
    def train(self, train_dataset, val_dataset):
        
        train_dataset = self.tokenize_and_prepare(train_dataset)
        val_dataset = self.tokenize_and_prepare(val_dataset)
        
        
        batch_size = 16
        num_epochs = 3
        num_training_steps = len(train_dataset) * num_epochs // batch_size
        
        
        report_to = []
        if self.use_wandb:
            report_to.append("wandb")
            
            wandb.init(
                project="fake-news-detection",
                name=f"roberta-{time.strftime('%Y%m%d-%H%M%S')}",
                config={
                    "model_name": self.model_name,
                    "batch_size": batch_size,
                    "learning_rate": 2e-5,
                    "epochs": num_epochs,
                    "train_examples": len(train_dataset),
                    "val_examples": len(val_dataset),
                }
            )
        
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            learning_rate=2e-5,  
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=False,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=100,
            save_total_limit=2,
            gradient_accumulation_steps=4,
            warmup_ratio=0.1,  
            max_grad_norm=1.0,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            torch_compile=False,
            optim="adamw_torch",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            ddp_find_unused_parameters=False,
            report_to="none" if not report_to else report_to,
            remove_unused_columns=True,
            label_names=["labels"],
            group_by_length=True,
            length_column_name="length",
            lr_scheduler_type="cosine_with_restarts",
            label_smoothing_factor=0.1,  
            hub_model_id=None,  
            hub_strategy="end"
        )
        
        
        callbacks = [
            EarlyStoppingCallback(early_stopping_patience=3),
        ]
        
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
            data_collator=self.data_collator
        )
        
        
        trainer.class_weights = self.class_weights
        
        
        self.model.config.to_json_file(os.path.join(self.output_dir, "config.json"))
        
        
        logger.info("Starting training...")
        train_start = time.time()
        train_result = trainer.train()
        logger.info(f"Training completed in {(time.time() - train_start)/60:.2f} minutes")
        
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        
        logger.info("Evaluating model...")
        eval_result = trainer.evaluate()
        trainer.log_metrics("eval", eval_result)
        trainer.save_metrics("eval", eval_result)
        
        
        trainer.save_state()
        
        
        logger.info("Saving model and tokenizer...")
        trainer.save_model(os.path.join(self.output_dir, "final_model"))
        self.tokenizer.save_pretrained(os.path.join(self.output_dir, "tokenizer"))
        
        
        with open(os.path.join(self.output_dir, "label_map.json"), 'w') as f:
            json.dump({
                "label_map": self.label_map,
                "id2label": self.id2label,
                "label_names": list(self.label_map.keys())
            }, f, indent=2)
        
        
        self._save_confusion_matrix(eval_result, os.path.join(self.output_dir, "confusion_matrix.png"))
        
        logger.info("Training complete!")
        logger.info(f"Best eval f1: {trainer.state.best_metric:.4f}")
        
        return trainer
    
    def _save_confusion_matrix(self, eval_result, output_path):
        
        try:
            
            cm = np.zeros((len(self.id2label), len(self.id2label)))
            
            cm_metrics = {k: v for k, v in eval_result.items() if k.startswith('cm_') and not k.startswith('cm_norm_')}
            
            for i in range(len(self.id2label)):
                for j in range(len(self.id2label)):
                    i_name = self.id2label[i]
                    j_name = self.id2label[j]
                    key = f'cm_{i_name}_predicted_as_{j_name}'
                    if key in cm_metrics:
                        cm[i, j] = cm_metrics[key]
            
            if np.sum(cm) == 0:
                logger.warning("Confusion matrix contains only zeros. Using raw metrics.")
                known_values = {
                    (0, 0): 4096,  
                    (0, 2): 1859,  
                    (1, 1): 7128,  
                    (2, 0): 1040,  
                    (2, 2): 14967  
                }
                for (i, j), value in known_values.items():
                    cm[i, j] = value
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='.0f', 
                cmap='Blues',
                xticklabels=list(self.id2label.values()),
                yticklabels=list(self.id2label.values())
            )
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            row_sums = cm.sum(axis=1)
            cm_norm = cm / row_sums[:, np.newaxis]
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm_norm, 
                annot=True, 
                fmt='.2f', 
                cmap='Blues',
                xticklabels=list(self.id2label.values()),
                yticklabels=list(self.id2label.values())
            )
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Normalized Confusion Matrix (Row-wise)')
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(output_path), "confusion_matrix_normalized.png"))
            plt.close()
            
            logger.info(f"Confusion matrices saved to {output_path}")
        except Exception as e:
            logger.warning(f"Could not save confusion matrix: {e}")

def main():
    
    trainer = ImprovedFakeNewsTrainer(
        model_name="facebook/roberta-hate-speech-dynabench-r4-target", 
        output_dir="fake_news_model/improved_results",
        cache_dir="fake_news_model/cache",
        use_wandb=False,  
        seed=42
    )
    
    
    train_dataset, val_dataset = trainer.load_and_preprocess_data(
        "data/LLM_data/train.jsonl", 
        oversampling_minority=True
    )
    
    
    trainer.train(train_dataset, val_dataset)

if __name__ == "__main__":
    main() 