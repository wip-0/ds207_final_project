""" Model Evaluation Functions"""

# importing scoring metrics
from sklearn.metrics import (
    classification_report, accuracy_score,
    recall_score, precision_score,
    f1_score, roc_auc_score
)


def evaluate_predictions(true_values, predictions, print_class_report = True):
    """
    Evaluate the performance of classification predictions based on the given true
    values. This function calculates recall, precision, F1-score, and optionally
    prints the classification report.

    The metrics are computed using the 'readmitted' column of the `true_values` data
    and the predicted values provided. This function can be used to assess the
    effectiveness of a classification model.

    :param true_values: A DataFrame containing the actual class labels. The column
        'readmitted' is used as the target data.
    :type true_values: pandas.DataFrame

    :param predictions: The predicted labels that correspond to the 'readmitted' target
        column in `true_values`.
    :type predictions: ndarray or list

    :param print_class_report: A boolean flag indicating whether to print the detailed
        classification report. Defaults to True.
    :type print_class_report: bool

    :return: A tuple containing the following calculated metrics:
        - recall: The recall score of the predictions.
        - precision: The precision score of the predictions.
        - f1: The F1-score of the predictions.
        - acc: The accuracy score of the predictions.
        - cl_report: The full classification report as a string.
    :rtype: tuple[float, float, float, str]
    """

    ##  sklearn-metrics ##
    # recall: TP / (TP + FN)
    recall = recall_score(
        y_true= true_values['readmitted'],
        y_pred= predictions)
    # precision: TP / (TP + FP)
    precision = precision_score(
        y_true= true_values['readmitted'],
        y_pred= predictions)
    # F1-score:
    # 2 * (precision * recall) / (precision + recall)
    f1 = f1_score(
        y_true= true_values['readmitted'],
        y_pred= predictions)
    # accuracy: (TP + TN) / (TP + TN + FP + FN)
    acc = accuracy_score(
        y_true=true_values["readmitted"],
        y_pred=predictions)
    # Classification report:
    cl_report = classification_report(
        y_true= true_values['readmitted'],
        y_pred= predictions,
        zero_division=0
    )
    if print_class_report:
        print(cl_report)

    return recall, precision, f1, acc,  cl_report



def build_results_row(split,
                      true_values,
                      predictions,
                      features,
                      best_parameters,
                      best_model,
                      model_name='logistic_regression'
                      ):
    """
    Builds a single result row dictionary containing performance metrics, model information,
    and other relevant details for a classification model.

    This function calculates various performance metrics for a classifier, including accuracy,
    recall, precision, F1-score, and ROC-AUC. It takes in model predictions, true target values,
    and relevant metadata to create a structured dictionary. The function also evaluates the model
    probability predictions for computing the ROC-AUC score.

    :param split: Indicates the data split (e.g., train, validation, test).
    :type split: str
    :param true_values: A DataFrame containing the true target values. It must include a column named 'readmitted'.
    :type true_values: pandas.DataFrame
    :param predictions: Model predictions for the target variable.
    :type predictions: numpy.ndarray or list
    :param features: Feature data used by the model for predictions.
    :type features: pandas.DataFrame or numpy.ndarray
    :param best_parameters: A dictionary containing the best parameters found during model optimization.
    :type best_parameters: dict
    :param best_model: The trained machine learning model that implements `predict_proba`.
    :type best_model: Any
    :param model_name: A string that represents the name of the model. Defaults to 'logistic_regression'.
    :type model_name: str
    :return: A dictionary containing the model name, split, number of rows, best parameters,
             and various classification metrics including accuracy, recall, precision, F1-score,
             and ROC-AUC.
    :rtype: dict
    """
    y_true = true_values['readmitted']
    y_proba = best_model.predict_proba(features)[:, 1]

    return {
        'model': model_name,
        'split': split,
        'n_rows': len(y_true),
        'best_parameters': best_parameters,
        'accuracy': accuracy_score(y_true, predictions),
        'recall': recall_score(y_true, predictions),
        'precision': precision_score(y_true, predictions),
        'f1': f1_score(y_true, predictions),
        'roc_auc': roc_auc_score(y_true, y_proba),
    }