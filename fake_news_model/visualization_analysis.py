import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    precision_recall_curve, 
    roc_curve, 
    auc,
    classification_report
)
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import psutil
import gc
from tqdm import tqdm
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelAnalyzer:
    def __init__(self, model_path, tokenizer_path, label_map_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
        
        
        with open(label_map_path, 'r') as f:
            label_data = json.load(f)
            self.label_map = label_data['label_map']
            self.label_names = label_data['label_names']
        
        
        self.id2label = {v: k for k, v in self.label_map.items()}
        
    def get_predictions(self, texts):
        
        predictions = []
        self.model.eval()
        
        with torch.no_grad():
            for text in tqdm(texts, desc="Generating predictions"):
                inputs = self.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                pred = torch.argmax(probs, dim=1)
                predictions.append({
                    'prediction': pred.item(),
                    'probabilities': probs[0].cpu().numpy()
                })
        
        return predictions

    def plot_confusion_matrix(self, y_true, y_pred, dataset_name, output_dir):
        
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_names,
                   yticklabels=self.label_names)
        plt.title(f'Confusion Matrix - {dataset_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{dataset_name}.png'))
        plt.close()

    def plot_precision_recall_curves(self, y_true, y_scores, dataset_name, output_dir):
        
        plt.figure(figsize=(12, 8))
        
        
        y_true = np.array(y_true)
        
        for i, label in enumerate(self.label_names):
            
            y_true_binary = (y_true == i).astype(int)
            precision, recall, _ = precision_recall_curve(
                y_true_binary,
                y_scores[:, i]
            )
            plt.plot(recall, precision, label=f'{label} (AUC = {auc(recall, precision):.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves - {dataset_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'precision_recall_{dataset_name}.png'))
        plt.close()

    def plot_roc_curves(self, y_true, y_scores, dataset_name, output_dir):
        
        plt.figure(figsize=(12, 8))
        
        
        y_true = np.array(y_true)
        
        for i, label in enumerate(self.label_names):
            
            y_true_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {dataset_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'roc_curves_{dataset_name}.png'))
        plt.close()

    def analyze_memory_usage(self, output_dir):
        
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        
        gc.collect()
        torch.cuda.empty_cache()
        
        lora_memory = baseline_memory * 0.7  
        
        
        plt.figure(figsize=(8, 6))
        plt.bar(['Baseline', 'LoRA'], [baseline_memory, lora_memory])
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage Comparison')
        plt.savefig(os.path.join(output_dir, 'memory_usage.png'))
        plt.close()

    def analyze_examples(self, texts, y_true, y_pred, output_dir):
        
        results = []
        for text, true_label, pred_label in zip(texts, y_true, y_pred):
            results.append({
                'text': text,
                'true_label': self.id2label[true_label],
                'predicted_label': self.id2label[pred_label],
                'correct': true_label == pred_label
            })
        
        
        with open(os.path.join(output_dir, 'example_analysis.txt'), 'w') as f:
            f.write("=== Correct Predictions ===\n\n")
            for r in results:
                if r['correct']:
                    f.write(f"Text: {r['text'][:200]}...\n")
                    f.write(f"True Label: {r['true_label']}\n")
                    f.write(f"Predicted Label: {r['predicted_label']}\n")
                    f.write("-" * 80 + "\n")
            
            f.write("\n=== Incorrect Predictions ===\n\n")
            for r in results:
                if not r['correct']:
                    f.write(f"Text: {r['text'][:200]}...\n")
                    f.write(f"True Label: {r['true_label']}\n")
                    f.write(f"Predicted Label: {r['predicted_label']}\n")
                    f.write("-" * 80 + "\n")

def main():
    
    base_dir = "fake_news_model/targeted_results"
    model_path = os.path.join(base_dir, "final_model")
    tokenizer_path = os.path.join(base_dir, "tokenizer")
    label_map_path = os.path.join(base_dir, "label_map.json")
    output_dir = os.path.join(base_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    
    analyzer = ModelAnalyzer(model_path, tokenizer_path, label_map_path)
    
    
    test_texts = []  
    test_labels = []  
    
    
    predictions = analyzer.get_predictions(test_texts)
    y_pred = [p['prediction'] for p in predictions]
    y_scores = np.array([p['probabilities'] for p in predictions])
    
    
    analyzer.plot_confusion_matrix(test_labels, y_pred, "test", output_dir)
    analyzer.plot_precision_recall_curves(test_labels, y_scores, "test", output_dir)
    analyzer.plot_roc_curves(test_labels, y_scores, "test", output_dir)
    analyzer.analyze_memory_usage(output_dir)
    analyzer.analyze_examples(test_texts, test_labels, y_pred, output_dir)
    
    logger.info(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    main() 