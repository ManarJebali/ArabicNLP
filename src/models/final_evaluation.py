"""
Script for comprehensive model evaluation and final reporting.
Usage: python final_evaluation.py
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_curve, auc,
    roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_model_and_data():
    """Load best model and test data."""
    print("="*80)
    print("LOADING MODEL AND DATA")
    print("="*80)
    
    # Load best model
    model_path = Path('results/models/best_model.pkl')
    with open(model_path, 'rb') as f:
        best_model = pickle.load(f)
    
    # Load metadata
    with open('results/models/model_results.json', 'r', encoding='utf-8') as f:
        model_metadata = json.load(f)
    
    print(f"\nBest Model: {model_metadata['best_model']}")
    print(f"F1 Score: {model_metadata['best_f1_score']:.4f}")
    print(f"Accuracy: {model_metadata['best_accuracy']:.4f}")
    
    # Load test data
    X_test = np.load('data/processed/X_test_features.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    # Load original test data for text analysis
    test_df = pd.read_csv('data/processed/test.csv')
    
    print(f"\nTest set shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y_test))}")
    
    return best_model, model_metadata, X_test, y_test, test_df


def generate_predictions(model, X_test, y_test, test_df):
    """Generate predictions and probabilities."""
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS")
    print("="*80)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get prediction probabilities if available
    has_proba = hasattr(model, 'predict_proba')
    if has_proba:
        y_pred_proba = model.predict_proba(X_test)
        print("✓ Probability predictions available")
    else:
        y_pred_proba = None
        print("✗ Model doesn't support probability predictions")
    
    # Add predictions to dataframe
    test_df['predicted_label'] = y_pred
    test_df['correct'] = (y_test == y_pred)
    
    if has_proba:
        test_df['confidence'] = y_pred_proba.max(axis=1)
    
    print(f"\nTotal predictions: {len(y_pred)}")
    print(f"Correct predictions: {test_df['correct'].sum()} ({test_df['correct'].mean()*100:.2f}%)")
    print(f"Incorrect predictions: {(~test_df['correct']).sum()} ({(~test_df['correct']).mean()*100:.2f}%)")
    
    return y_pred, y_pred_proba, has_proba, test_df


def comprehensive_metrics(y_test, y_pred):
    """Calculate and display comprehensive metrics."""
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(y_test, y_pred, digits=4))
    
    # Per-class metrics
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    classes = [k for k in report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame([
        {
            'Class': cls,
            'Precision': report_dict[cls]['precision'],
            'Recall': report_dict[cls]['recall'],
            'F1-Score': report_dict[cls]['f1-score'],
            'Support': int(report_dict[cls]['support'])
        }
        for cls in classes
    ])
    
    print("\n" + "="*80)
    print("PER-CLASS METRICS")
    print("="*80)
    print(metrics_df.to_string(index=False))
    
    return metrics_df, report_dict


def visualize_per_class_metrics(metrics_df):
    """Visualize per-class performance."""
    print("\n✓ Creating per-class metrics visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Precision
    metrics_df.plot(x='Class', y='Precision', kind='bar', ax=axes[0, 0], legend=False)
    axes[0, 0].set_title('Precision by Class')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].axhline(y=metrics_df['Precision'].mean(), color='r', linestyle='--', label='Mean')
    axes[0, 0].legend()
    
    # Recall
    metrics_df.plot(x='Class', y='Recall', kind='bar', ax=axes[0, 1], legend=False, color='orange')
    axes[0, 1].set_title('Recall by Class')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].axhline(y=metrics_df['Recall'].mean(), color='r', linestyle='--', label='Mean')
    axes[0, 1].legend()
    
    # F1-Score
    metrics_df.plot(x='Class', y='F1-Score', kind='bar', ax=axes[1, 0], legend=False, color='green')
    axes[1, 0].set_title('F1-Score by Class')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].axhline(y=metrics_df['F1-Score'].mean(), color='r', linestyle='--', label='Mean')
    axes[1, 0].legend()
    
    # Support
    metrics_df.plot(x='Class', y='Support', kind='bar', ax=axes[1, 1], legend=False, color='purple')
    axes[1, 1].set_title('Support (Number of Samples) by Class')
    axes[1, 1].set_ylabel('Number of Samples')
    
    plt.tight_layout()
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('results/figures/per_class_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved to results/figures/per_class_metrics.png")


def analyze_confusion_matrix(y_test, y_pred):
    """Compute and visualize confusion matrix."""
    print("\n" + "="*80)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*80)
    
    # Compute confusion matrices
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')
    
    # Plot confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    axes[0].set_title('Confusion Matrix (Counts)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    axes[1].set_title('Confusion Matrix (Normalized by True Label)')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('results/figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Confusion matrix saved to results/figures/confusion_matrix.png")
    
    # Analyze misclassifications
    print("\n" + "="*80)
    print("MOST COMMON MISCLASSIFICATIONS")
    print("="*80)
    
    misclass = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > 0:
                misclass.append({
                    'True Label': i,
                    'Predicted Label': j,
                    'Count': cm[i][j],
                    'Percentage': f"{100 * cm_normalized[i][j]:.1f}%"
                })
    
    misclass_df = pd.DataFrame(misclass).sort_values('Count', ascending=False)
    print(misclass_df.head(10).to_string(index=False))
    
    return cm, cm_normalized, misclass_df


def error_analysis(test_df, y_test):
    """Analyze misclassified examples."""
    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80)
    
    # Get misclassified examples
    misclassified = test_df[~test_df['correct']].copy()
    print(f"\nTotal misclassified samples: {len(misclassified)}")
    print(f"Misclassification rate: {len(misclassified) / len(test_df) * 100:.2f}%")
    
    # Sample misclassified examples
    print("\n" + "="*80)
    print("SAMPLE MISCLASSIFIED EXAMPLES")
    print("="*80)
    
    n_samples = min(10, len(misclassified))
    for idx, row in misclassified.sample(n_samples).iterrows():
        print(f"\nText: {row['text'][:100]}...")
        print(f"True Label: {row['label']}")
        print(f"Predicted Label: {row['predicted_label']}")
        print("-" * 80)
    
    return misclassified


def confidence_analysis(test_df, has_proba):
    """Analyze prediction confidence."""
    if not has_proba:
        print("\n✗ Confidence analysis not available (no probabilities)")
        return None, None
    
    print("\n" + "="*80)
    print("CONFIDENCE ANALYSIS")
    print("="*80)
    
    # Compare confidence for correct vs incorrect predictions
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Distribution
    test_df[test_df['correct']]['confidence'].hist(
        bins=30, alpha=0.7, label='Correct', ax=axes[0]
    )
    test_df[~test_df['correct']]['confidence'].hist(
        bins=30, alpha=0.7, label='Incorrect', ax=axes[0]
    )
    axes[0].set_xlabel('Prediction Confidence')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Confidence Distribution: Correct vs Incorrect')
    axes[0].legend()
    
    # Box plot
    test_df.boxplot(column='confidence', by='correct', ax=axes[1])
    axes[1].set_xlabel('Correct Prediction')
    axes[1].set_ylabel('Confidence')
    axes[1].set_title('Confidence by Prediction Correctness')
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig('results/figures/confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Confidence analysis saved to results/figures/confidence_analysis.png")
    
    # Statistics
    print(f"\nMean confidence (correct): {test_df[test_df['correct']]['confidence'].mean():.4f}")
    print(f"Mean confidence (incorrect): {test_df[~test_df['correct']]['confidence'].mean():.4f}")
    
    # Low confidence correct predictions
    low_conf_correct = test_df[(test_df['correct']) & (test_df['confidence'] < 0.6)]
    print(f"\nLow confidence correct predictions: {len(low_conf_correct)}")
    
    # High confidence incorrect predictions
    high_conf_incorrect = test_df[(~test_df['correct']) & (test_df['confidence'] > 0.8)]
    print(f"High confidence incorrect predictions: {len(high_conf_incorrect)}")
    
    return low_conf_correct, high_conf_incorrect


def feature_importance_analysis(model):
    """Analyze feature importance if available."""
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    # Feature importance for tree-based models
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Get top N important features
        top_n = 20
        indices = np.argsort(importances)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [f"Feature {i}" for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features')
        plt.tight_layout()
        plt.savefig('results/figures/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Feature importance saved to results/figures/feature_importance.png")
        
        print(f"\nTop {top_n} most important features:")
        for idx, imp in zip(indices[::-1], importances[indices][::-1]):
            print(f"  Feature {idx}: {imp:.6f}")
    
    # Coefficients for linear models
    elif hasattr(model, 'coef_'):
        coefs = np.abs(model.coef_).mean(axis=0)
        
        top_n = 20
        indices = np.argsort(coefs)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), coefs[indices])
        plt.yticks(range(top_n), [f"Feature {i}" for i in indices])
        plt.xlabel('|Coefficient|')
        plt.title(f'Top {top_n} Features by Coefficient Magnitude')
        plt.tight_layout()
        plt.savefig('results/figures/feature_coefficients.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Feature coefficients saved to results/figures/feature_coefficients.png")
    else:
        print("\n✗ Model doesn't provide feature importance or coefficients")


def roc_curve_analysis(y_test, y_pred_proba, has_proba):
    """Generate ROC curves for multi-class classification."""
    if not has_proba or len(np.unique(y_test)) <= 2:
        print("\n✗ ROC curve analysis not available")
        return
    
    print("\n" + "="*80)
    print("ROC CURVE ANALYSIS")
    print("="*80)
    
    # Binarize labels for multi-class ROC
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_bin.shape[1]
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Multi-class Classification')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ ROC curves saved to results/figures/roc_curves.png")
    
    print("\n" + "="*80)
    print("AUC SCORES BY CLASS")
    print("="*80)
    for i in range(n_classes):
        print(f"Class {i}: {roc_auc[i]:.4f}")
    print(f"\nMacro-average AUC: {np.mean(list(roc_auc.values())):.4f}")


def model_comparison_summary(model_metadata):
    """Display final model comparison."""
    print("\n" + "="*80)
    print("FINAL MODEL COMPARISON")
    print("="*80)
    
    all_models_performance = pd.DataFrame([
        {
            'Model': name,
            'Accuracy': data['accuracy'],
            'Precision': data['precision'],
            'Recall': data['recall'],
            'F1 Score': data['f1']
        }
        for name, data in model_metadata['all_models'].items()
    ]).sort_values('F1 Score', ascending=False)
    
    print(all_models_performance.to_string(index=False))
    
    # Visualize final comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    all_models_performance.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1 Score']].plot(
        kind='bar', ax=ax
    )
    ax.set_title('Final Model Performance Comparison')
    ax.set_ylabel('Score')
    ax.set_ylim([0, 1])
    ax.legend(loc='lower right')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/figures/final_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Model comparison saved to results/figures/final_model_comparison.png")
    
    return all_models_performance


def generate_final_report(model_metadata, y_test, y_pred, metrics_df, misclass_df, 
                          misclassified, test_df, has_proba, low_conf_correct, high_conf_incorrect):
    """Generate comprehensive final report."""
    print("\n" + "="*80)
    print("GENERATING FINAL REPORT")
    print("="*80)
    
    report_lines = [
        "="*80,
        "ARABIC NLP TEXT CLASSIFICATION - FINAL REPORT",
        "="*80,
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "="*80,
        "1. DATASET SUMMARY",
        "="*80,
        f"Training samples: {len(np.load('data/processed/y_train.npy'))}",
        f"Test samples: {len(y_test)}",
        f"Number of classes: {len(np.unique(y_test))}",
        "",
        "="*80,
        "2. BEST MODEL",
        "="*80,
        f"Model: {model_metadata['best_model']}",
        f"Accuracy: {model_metadata['best_accuracy']:.4f}",
        f"F1 Score: {model_metadata['best_f1_score']:.4f}",
        "",
        "="*80,
        "3. DETAILED CLASSIFICATION REPORT",
        "="*80,
        classification_report(y_test, y_pred),
        "",
        "="*80,
        "4. PER-CLASS METRICS",
        "="*80,
        metrics_df.to_string(index=False),
        "",
        "="*80,
        "5. TOP MISCLASSIFICATIONS",
        "="*80,
        misclass_df.head(10).to_string(index=False),
        "",
        "="*80,
        "6. ERROR ANALYSIS",
        "="*80,
        f"Total misclassified: {len(misclassified)} ({len(misclassified)/len(test_df)*100:.2f}%)",
        f"Correctly classified: {test_df['correct'].sum()} ({test_df['correct'].mean()*100:.2f}%)",
        "",
    ]
    
    if has_proba:
        report_lines.extend([
            "="*80,
            "7. CONFIDENCE ANALYSIS",
            "="*80,
            f"Mean confidence (correct): {test_df[test_df['correct']]['confidence'].mean():.4f}",
            f"Mean confidence (incorrect): {test_df[~test_df['correct']]['confidence'].mean():.4f}",
            f"Low confidence correct: {len(low_conf_correct)}",
            f"High confidence incorrect: {len(high_conf_incorrect)}",
            ""
        ])
    
    # Add insights
    best_class = metrics_df.loc[metrics_df['F1-Score'].idxmax()]
    worst_class = metrics_df.loc[metrics_df['F1-Score'].idxmin()]
    
    report_lines.extend([
        "="*80,
        "8. KEY INSIGHTS",
        "="*80,
        f"\nBest Performing Class: {best_class['Class']} (F1={best_class['F1-Score']:.4f})",
        f"Worst Performing Class: {worst_class['Class']} (F1={worst_class['F1-Score']:.4f})",
        f"Overall Performance: Accuracy={model_metadata['best_accuracy']:.4f}, F1={model_metadata['best_f1_score']:.4f}",
        "",
        "="*80,
        "9. RECOMMENDATIONS",
        "="*80,
        "1. Focus on improving worst-performing class through:",
        "   - Additional training data collection",
        "   - Feature engineering specific to this class",
        "   - Class-specific preprocessing",
        "",
        "2. Investigate high-confidence errors for:",
        "   - Data labeling issues",
        "   - Ambiguous cases",
        "   - Model bias",
        "",
        "3. Consider ensemble methods to boost performance",
        "",
        "4. Deploy model with confidence thresholds for production",
        "",
        "5. Monitor model performance on new data for drift",
        "",
        "="*80,
        "END OF REPORT",
        "="*80
    ])
    
    # Save report
    report_path = Path('results/reports')
    report_path.mkdir(parents=True, exist_ok=True)
    
    with open(report_path / 'final_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n✓ Report saved to: {report_path / 'final_report.txt'}")
    
    # Save predictions
    test_df[['text', 'label', 'predicted_label', 'correct']].to_csv(
        report_path / 'test_predictions.csv', index=False, encoding='utf-8'
    )
    print(f"✓ Predictions saved to: {report_path / 'test_predictions.csv'}")


def save_production_results(model_metadata, y_test, metrics_df):
    """Save production-ready package."""
    print("\n" + "="*80)
    print("SAVING PRODUCTION RESULTS")
    print("="*80)
    
    production_results = {
        'model_name': model_metadata['best_model'],
        'model_path': 'results/models/best_model.pkl',
        'test_accuracy': float(model_metadata['best_accuracy']),
        'test_f1_score': float(model_metadata['best_f1_score']),
        'num_features': int(np.load('data/processed/X_test_features.npy').shape[1]),
        'num_classes': int(len(np.unique(y_test))),
        'class_labels': [int(x) for x in np.unique(y_test)],
        'per_class_metrics': metrics_df.to_dict('records'),
        'evaluation_date': datetime.now().isoformat(),
        'dataset_info': {
            'train_size': int(len(np.load('data/processed/y_train.npy'))),
            'test_size': int(len(y_test))
        }
    }
    
    report_path = Path('results/reports')
    with open(report_path / 'production_results.json', 'w', encoding='utf-8') as f:
        json.dump(production_results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Production results saved to: {report_path / 'production_results.json'}")


def main():
    """Main evaluation pipeline."""
    print("\n" + "="*80)
    print("ARABIC NLP - FINAL EVALUATION PIPELINE")
    print("="*80)
    
    # Load model and data
    best_model, model_metadata, X_test, y_test, test_df = load_model_and_data()
    
    # Generate predictions
    y_pred, y_pred_proba, has_proba, test_df = generate_predictions(
        best_model, X_test, y_test, test_df
    )
    
    # Comprehensive metrics
    metrics_df, report_dict = comprehensive_metrics(y_test, y_pred)
    visualize_per_class_metrics(metrics_df)
    
    # Confusion matrix analysis
    cm, cm_normalized, misclass_df = analyze_confusion_matrix(y_test, y_pred)
    
    # Error analysis
    misclassified = error_analysis(test_df, y_test)
    
    # Confidence analysis
    low_conf_correct, high_conf_incorrect = confidence_analysis(test_df, has_proba)
    
    # Feature importance
    feature_importance_analysis(best_model)
    
    # ROC curves
    roc_curve_analysis(y_test, y_pred_proba, has_proba)
    
    # Model comparison
    all_models_performance = model_comparison_summary(model_metadata)
    
    # Generate final report
    generate_final_report(
        model_metadata, y_test, y_pred, metrics_df, misclass_df,
        misclassified, test_df, has_proba, low_conf_correct, high_conf_incorrect
    )
    
    # Save production results
    save_production_results(model_metadata, y_test, metrics_df)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nBest Model: {model_metadata['best_model']}")
    print(f"Test Accuracy: {model_metadata['best_accuracy']:.4f}")
    print(f"Test F1 Score: {model_metadata['best_f1_score']:.4f}")
    print("\nAll results saved to results/")
    print("  - Figures: results/figures/")
    print("  - Reports: results/reports/")


if __name__ == "__main__":
    main()
