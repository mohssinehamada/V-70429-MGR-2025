import os
import json
from visualization_analysis import ModelAnalyzer
import logging
import numpy as np
import pickle


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_data(data_path, label_map):
    
    texts = []
    labels = []
    
    with open(data_path, 'r') as f:
        for line in f:
            try:
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
                
                
                if label in label_map:
                    texts.append(text)
                    labels.append(label_map[label])
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line: {line[:100]}...")
                continue
    
    logger.info(f"Loaded {len(texts)} examples")
    logger.info(f"Label distribution: {np.bincount(labels)}")
    return texts, labels

def load_or_generate_predictions(analyzer, texts, cache_path):
    
    if os.path.exists(cache_path):
        logger.info("Loading cached predictions...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    logger.info("Generating new predictions...")
    predictions = analyzer.get_predictions(texts)
    
    
    logger.info("Caching predictions...")
    with open(cache_path, 'wb') as f:
        pickle.dump(predictions, f)
    
    return predictions

def main():
    
    base_dir = "fake_news_model/targeted_results"
    model_path = os.path.join(base_dir, "final_model")
    tokenizer_path = os.path.join(base_dir, "tokenizer")
    label_map_path = os.path.join(base_dir, "label_map.json")
    output_dir = os.path.join(base_dir, "visualizations")
    cache_dir = os.path.join(base_dir, "cache")
    test_data_path = "data/LLM_data/train.jsonl"
    
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    
    with open(label_map_path, 'r') as f:
        label_data = json.load(f)
        label_map = label_data['label_map']
    
    
    logger.info("Loading test data...")
    test_texts, test_labels = load_test_data(test_data_path, label_map)
    
    if len(test_texts) == 0:
        logger.error("No valid data loaded. Please check the data format and label mapping.")
        return
    
    
    logger.info("Initializing model analyzer...")
    analyzer = ModelAnalyzer(model_path, tokenizer_path, label_map_path)
    
    
    cache_path = os.path.join(cache_dir, "predictions.pkl")
    predictions = load_or_generate_predictions(analyzer, test_texts, cache_path)
    
    y_pred = [p['prediction'] for p in predictions]
    y_scores = np.array([p['probabilities'] for p in predictions])
    
    
    logger.info("Generating visualizations...")
    analyzer.plot_confusion_matrix(test_labels, y_pred, "test", output_dir)
    analyzer.plot_precision_recall_curves(test_labels, y_scores, "test", output_dir)
    analyzer.plot_roc_curves(test_labels, y_scores, "test", output_dir)
    analyzer.analyze_memory_usage(output_dir)
    analyzer.analyze_examples(test_texts, test_labels, y_pred, output_dir)
    
    logger.info(f"Analysis complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main() 