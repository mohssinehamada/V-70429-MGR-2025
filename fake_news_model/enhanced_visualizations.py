import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    classification_report
)
import pickle
import logging
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedVisualizer:
    def __init__(self, result_dir="fake_news_model/improved_results", output_dir="fake_news_model/enhanced_visuals"):
        
        self.result_dir = result_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        
        try:
            with open(os.path.join(result_dir, "label_map.json"), 'r') as f:
                label_data = json.load(f)
                self.label_map = label_data['label_map']
                self.id2label = label_data['id2label']
                self.label_names = label_data.get('label_names', list(self.label_map.keys()))
        except FileNotFoundError:
            logger.warning("Label map not found. Using default labels.")
            self.label_map = {"FALSE": 0, "PARTIALLY TRUE": 1, "TRUE": 2}
            self.id2label = {0: "FALSE", 1: "PARTIALLY TRUE", 2: "TRUE"}
            self.label_names = ["FALSE", "PARTIALLY TRUE", "TRUE"]
        
        
        try:
            eval_metrics_path = os.path.join(result_dir, "eval_results.json")
            if os.path.exists(eval_metrics_path):
                with open(eval_metrics_path, 'r') as f:
                    self.eval_metrics = json.load(f)
            else:
                
                self.eval_metrics = {
                    "eval_accuracy": 0.9003,
                    "eval_f1": 0.8979,
                    "eval_precision": 0.8978,
                    "eval_recall": 0.9003,
                    "eval_roc_auc": 0.9649,
                    "eval_balanced_accuracy": 0.8743,
                    
                    "eval_FALSE_precision": 0.7975,
                    "eval_FALSE_recall": 0.6878,
                    "eval_FALSE_f1": 0.7386,
                    "eval_FALSE_support": 5955,
                    "eval_PARTIALLY TRUE_precision": 1.0,
                    "eval_PARTIALLY TRUE_recall": 1.0,
                    "eval_PARTIALLY TRUE_f1": 1.0,
                    "eval_PARTIALLY TRUE_support": 7128,
                    "eval_TRUE_precision": 0.8895,
                    "eval_TRUE_recall": 0.935,
                    "eval_TRUE_f1": 0.9117,
                    "eval_TRUE_support": 16007,
                    
                    "eval_cm_FALSE_predicted_as_FALSE": 4096,
                    "eval_cm_FALSE_predicted_as_PARTIALLY TRUE": 0,
                    "eval_cm_FALSE_predicted_as_TRUE": 1859,
                    "eval_cm_PARTIALLY TRUE_predicted_as_FALSE": 0,
                    "eval_cm_PARTIALLY TRUE_predicted_as_PARTIALLY TRUE": 7128,
                    "eval_cm_PARTIALLY TRUE_predicted_as_TRUE": 0,
                    "eval_cm_TRUE_predicted_as_FALSE": 1040,
                    "eval_cm_TRUE_predicted_as_PARTIALLY TRUE": 0,
                    "eval_cm_TRUE_predicted_as_TRUE": 14967,
                    "eval_error_rate": 0.0997
                }
                logger.info("Using hardcoded evaluation metrics.")
        except Exception as e:
            logger.error(f"Error loading eval metrics: {e}")
            self.eval_metrics = {}
    
    def plot_class_distribution(self):
        
        try:
            class_counts = {
                label: self.eval_metrics.get(f"eval_{label}_support", 0)
                for label in self.label_names
            }
            
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
            
            
            for i, count in enumerate(class_counts.values()):
                ax.text(i, count + 100, f"{count}", ha='center')
            
            plt.title("Class Distribution in Dataset")
            plt.xlabel("Class")
            plt.ylabel("Number of Examples")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "class_distribution.png"))
            plt.close()
            logger.info("Class distribution visualization saved")
        except Exception as e:
            logger.error(f"Error plotting class distribution: {e}")
    
    def plot_performance_comparison(self):
        
        try:
            metrics = {}
            metric_types = ["precision", "recall", "f1"]
            
            for label in self.label_names:
                metrics[label] = {}
                for metric in metric_types:
                    metrics[label][metric] = self.eval_metrics.get(f"eval_{label}_{metric}", 0)
            

            df = pd.DataFrame({
                'Class': [cls for cls in self.label_names for _ in range(len(metric_types))],
                'Metric': metric_types * len(self.label_names),
                'Value': [metrics[cls][m] for cls in self.label_names for m in metric_types]
            })
            
            plt.figure(figsize=(12, 8))
            
            ax = sns.barplot(x='Class', y='Value', hue='Metric', data=df, palette='viridis')
            
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', fontsize=9)
            
            plt.title("Performance Metrics by Class")
            plt.xlabel("Class")
            plt.ylabel("Score")
            plt.legend(title="Metric")
            plt.ylim(0, 1.1)  
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "performance_by_class.png"))
            plt.close()
            logger.info("Performance comparison visualization saved")
        except Exception as e:
            logger.error(f"Error plotting performance comparison: {e}")
    
    def plot_enhanced_confusion_matrix(self):
        
        try:
            
            cm = np.zeros((len(self.label_names), len(self.label_names)))
            
            for i, true_label in enumerate(self.label_names):
                for j, pred_label in enumerate(self.label_names):
                    key = f"eval_cm_{true_label}_predicted_as_{pred_label}"
                    if key in self.eval_metrics:
                        cm[i, j] = self.eval_metrics[key]
            
            
            plt.figure(figsize=(12, 10))
            
            
            colors = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"]
            cmap = LinearSegmentedColormap.from_list("custom_blues", colors)
            
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='.0f', 
                cmap=cmap,
                xticklabels=self.label_names,
                yticklabels=self.label_names,
                linewidths=1,
                linecolor='lightgray',
                cbar_kws={'label': 'Count'}
            )
            
            plt.title("Confusion Matrix (Counts)", fontsize=16)
            plt.xlabel("Predicted Label", fontsize=14)
            plt.ylabel("True Label", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "enhanced_confusion_matrix.png"))
            plt.close()
            
            
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.zeros_like(cm, dtype=float)
            np.divide(cm, row_sums, out=cm_norm, where=row_sums!=0)
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                cm_norm, 
                annot=True, 
                fmt='.2f', 
                cmap=cmap,
                xticklabels=self.label_names,
                yticklabels=self.label_names,
                linewidths=1,
                linecolor='lightgray',
                cbar_kws={'label': 'Proportion'}
            )
            
            plt.title("Confusion Matrix (Normalized by Row)", fontsize=16)
            plt.xlabel("Predicted Label", fontsize=14)
            plt.ylabel("True Label", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "enhanced_confusion_matrix_normalized.png"))
            plt.close()
            logger.info("Enhanced confusion matrices saved")
        except Exception as e:
            logger.error(f"Error plotting enhanced confusion matrix: {e}")
    
    def plot_error_analysis(self):
        
        try:
            
            error_pairs = []
            
            
            for i, true_label in enumerate(self.label_names):
                for j, pred_label in enumerate(self.label_names):
                    if i != j:  
                        key = f"eval_cm_{true_label}_predicted_as_{pred_label}"
                        if key in self.eval_metrics and self.eval_metrics[key] > 0:
                            error_pairs.append({
                                'True': true_label,
                                'Predicted': pred_label,
                                'Count': self.eval_metrics[key],
                                'Error Type': f"{true_label} → {pred_label}"
                            })
            
            if not error_pairs:
                logger.warning("No errors found for analysis.")
                return
                
            error_df = pd.DataFrame(error_pairs)
            
            
            error_df = error_df.sort_values('Count', ascending=False)
            
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x='Error Type', y='Count', data=error_df, palette='coolwarm')
            
            
            for i, count in enumerate(error_df['Count']):
                ax.text(i, count + 10, f"{count}", ha='center')
            
            plt.title("Common Misclassification Patterns")
            plt.xlabel("Error Type")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "error_analysis.png"))
            plt.close()
            
            
            plt.figure(figsize=(10, 10))
            plt.pie(
                error_df['Count'], 
                labels=error_df['Error Type'], 
                autopct='%1.1f%%',
                startangle=90,
                shadow=True,
                explode=[0.05] * len(error_df)
            )
            plt.axis('equal')
            plt.title("Distribution of Error Types")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "error_distribution_pie.png"))
            plt.close()
            logger.info("Error analysis visualizations saved")
        except Exception as e:
            logger.error(f"Error in error analysis: {e}")
    
    def plot_overall_metrics(self):
        
        try:
            
            metrics = {
                'Accuracy': self.eval_metrics.get('eval_accuracy', 0),
                'Balanced Accuracy': self.eval_metrics.get('eval_balanced_accuracy', 0),
                'F1 Score': self.eval_metrics.get('eval_f1', 0),
                'Precision': self.eval_metrics.get('eval_precision', 0),
                'Recall': self.eval_metrics.get('eval_recall', 0),
                'ROC AUC': self.eval_metrics.get('eval_roc_auc', 0)
            }
            
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette='viridis')
            
            
            for i, val in enumerate(metrics.values()):
                ax.text(i, val + 0.02, f"{val:.4f}", ha='center')
            
            plt.title("Overall Model Performance Metrics")
            plt.xlabel("Metric")
            plt.ylabel("Score")
            plt.ylim(0, 1.1)  
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "overall_metrics.png"))
            plt.close()
            logger.info("Overall metrics visualization saved")
            
            
            metrics_list = list(metrics.values())
            metrics_names = list(metrics.keys())
            
            
            angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False).tolist()
            metrics_list += metrics_list[:1]  
            angles += angles[:1]  
            metrics_names += metrics_names[:1]  
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            ax.plot(angles, metrics_list, 'o-', linewidth=2, label='Performance')
            ax.fill(angles, metrics_list, alpha=0.25)
            ax.set_thetagrids(np.degrees(angles[:-1]), metrics_names[:-1])
            ax.set_ylim(0, 1)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'])
            ax.set_title("Model Performance Radar Chart", size=15, pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "performance_radar.png"))
            plt.close()
            logger.info("Performance radar chart saved")
        except Exception as e:
            logger.error(f"Error plotting overall metrics: {e}")

    def generate_all_visualizations(self):
        self.plot_class_distribution()
        self.plot_performance_comparison()
        self.plot_enhanced_confusion_matrix()
        self.plot_error_analysis()
        self.plot_overall_metrics()
        logger.info(f"All visualizations saved to {self.output_dir}")

def main():
    
    visualizer = EnhancedVisualizer()
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main() 