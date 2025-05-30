import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pickle
from sklearn.metrics import classification_report, confusion_matrix

def analyze_confusion_matrix(cm, label_names):
    """Analyze confusion matrix and print detailed metrics"""
    print("\n=== Confusion Matrix Analysis ===\n")
    
    print("Raw Confusion Matrix:")
    print(cm)
    print("\n")
    
    total = cm.sum()
    print("Class-wise Analysis:")
    for i, label in enumerate(label_names):
        true_positives = cm[i, i]
        false_positives = cm[:, i].sum() - true_positives
        false_negatives = cm[i, :].sum() - true_positives
        true_negatives = total - (true_positives + false_positives + false_negatives)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{label}:")
        print(f"  True Positives: {true_positives}")
        print(f"  False Positives: {false_positives}")
        print(f"  False Negatives: {false_negatives}")
        print(f"  True Negatives: {true_negatives}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1 Score: {f1:.3f}")
    
    accuracy = np.diag(cm).sum() / total
    print(f"\nOverall Accuracy: {accuracy:.3f}")
    
    print("\nCommon Misclassifications:")
    for i, true_label in enumerate(label_names):
        for j, pred_label in enumerate(label_names):
            if i != j and cm[i, j] > 0:
                print(f"  {true_label} → {pred_label}: {cm[i, j]} examples")

def main():
    base_dir = "fake_news_model/targeted_results"
    cache_dir = os.path.join(base_dir, "cache")
    label_map_path = os.path.join(base_dir, "label_map.json")
    
    with open(label_map_path, 'r') as f:
        label_data = json.load(f)
        label_names = label_data['label_names']
    
    cache_path = os.path.join(cache_dir, "predictions.pkl")
    if not os.path.exists(cache_path):
        print("Error: No predictions found in cache. Please run run_analysis.py first.")
        return
    
    with open(cache_path, 'rb') as f:
        predictions = pickle.load(f)
    
    y_pred = [p['prediction'] for p in predictions]
    
    test_data_path = "data/LLM_data/train.jsonl"
    with open(test_data_path, 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    y_true = []
    for item in test_data:
        original_label = item.get('label', 'UNKNOWN')
        if original_label == 'SUPPORTS':
            label = 'TRUE'
        elif original_label == 'REFUTES':
            label = 'FALSE'
        else:
            label = 'PARTIALLY TRUE'
        if label in label_data['label_map']:
            y_true.append(label_data['label_map'][label])
    
    cm = confusion_matrix(y_true, y_pred)
    
    analyze_confusion_matrix(cm, label_names)

if __name__ == "__main__":
    main() 