"""Visualizations for model analysis"""

#### IMPORTS
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve,
                             roc_auc_score,
                             confusion_matrix,
                             precision_score,
                             recall_score)

##### FUNCTIONS

def plot_roc_curves_sklearn(models,
                            X_test,
                            y_test,
                            model_names=None
                            ):
    """
    This function plots ROC curves for the given models.
    Parameters:
        models: A list of trained sklearn models.
        X_test: The test set features.
        y_test: The test set targets.
        model_names: (optional) A list of names of the models.
    Returns:
        A dictionary of ROC AUC scores for the models.
    """
    # Initialize dictionary for AUC scores
    auc_scores = {}
    sns.set_theme(style="darkgrid", palette="bright")
    # Initialize the figure
    plt.figure(figsize=(10, 6))

    for model, name in zip(models, model_names):
        # Get the model's prediction probabilities
        y_score = model.predict_proba(X_test)[:, 1]

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_score)

        # Calculate the AUC score
        auc_score = roc_auc_score(y_test, y_score)
        auc_scores[name] = auc_score

        # Plot the ROC curve directly using plt.plot for every given model
        plt.plot(fpr, tpr, label=f'{name}, AUC={auc_score:.2f}')

    #
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # Place the legend outside the figure/plot
    plt.legend(title='Model', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()
    return auc_scores


def plot_roc_curves(y_test,
                    y_scores, # predicted probabilities
                    model_names=None,
                    title='ROC Curve'):
    """
    Plot ROC curves from true labels and predicted scores/probabilities.

    Parameters:
        y_test: True target values.
        y_scores: A list of predicted probabilities/scores for the positive class.
                  Each item should be a 1D array.
        model_names: Optional list of model names.
        title: Plot title.

    Returns:
        A dictionary of ROC AUC scores.
    """

    # If a single array of scores is provided, convert it to a list
    # so the rest of the function can handle one or multiple models consistently.
    if not isinstance(y_scores, list):
        y_scores = [y_scores]
    # If no model names are provided, create default names.
    if model_names is None:
        model_names = [f'Model {i + 1}' for i in range(len(y_scores))]
    # Initialize dictionary to store AUC scores for each model.
    auc_scores = {}
    # Set the plot style.
    sns.set_theme(style='darkgrid', palette='bright')
    # Initialize the figure.
    plt.figure(figsize=(10, 6))
    # Loop through each set of prediction scores
    # and its corresponding model name.
    for score, name in zip(y_scores, model_names):

        # Calculate false positive rate and true positive rate.
        fpr, tpr, _ = roc_curve(y_test, score)
        # Calculate the ROC AUC score.
        auc_score = roc_auc_score(y_test, score)
        # Store the AUC score in the dictionary.
        auc_scores[name] = auc_score
        # Plot the ROC curve for the current model.
        plt.plot(fpr, tpr, label=f'{name}, AUC={auc_score:.2f}')

    # Plot the diagonal reference line representing random guessing.
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
    # Add plot title and axis labels.
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # Place the legend outside the plot area.
    plt.legend(title='Model', bbox_to_anchor=(1, 1))
    # Add grid lines.
    plt.grid(True)
    # Plot it
    plt.tight_layout()
    plt.show()
    # Return the AUC scores.
    return auc_scores


def plot_confusion_matrix(y_true,
                          y_pred,
                          title='Confusion Matrix',
                          class_names=None,
                          average='binary',
                          fig_size = (6, 6)):
    """
    Plot a confusion matrix from true labels and predicted labels.

    Parameters:
        y_true: True target values.
        y_pred: Predicted target values.
        title: Plot title.
        class_names: Optional class labels for the matrix axes.
        average: Averaging method for precision/recall.
                 Use 'binary' for binary classification,
                 'macro' or 'weighted' for multiclass.
        fig_size: Figure size.
    """

    # Calculate precision and recall
    precision = precision_score(y_true, y_pred,
                                average=average,
                                zero_division=0)
    recall = recall_score(y_true, y_pred,
                          average=average,
                          zero_division=0)

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Generate default class names if not provided
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]

    ## Confusion matrix
    plt.figure(figsize= fig_size)
    # plot heatmap
    sns.heatmap( cm, annot=True, fmt='d',
                 cmap='Blues',
                 xticklabels=class_names,
                 yticklabels=class_names
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    # Add Precision and Recall formulas/scores as text in the matrix
    plt.text(1.55, 0.9,
              r'$Precision = \frac{TP}{TP + FP}$' + '\n' + f'Score: {precision:.2f}',
              horizontalalignment='center',
              verticalalignment='center',
              transform=plt.gca().transAxes)
    plt.text(1.55, 0.7,
             r'$Recall = \frac{TP}{TP + FN}$' + '\n' + f'Score: {recall:.2f}',
            horizontalalignment='center',
            verticalalignment='center',
            transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.show()

    #
    return {'precision': precision,
            'recall': recall,
            'confusion_matrix': cm }