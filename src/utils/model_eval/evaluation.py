""" Model Evaluation Functions"""

# importing various scoring metrics
from sklearn.metrics import (classification_report,
                             recall_score,
                             precision_score,
                             f1_score)


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
    # Classification report:
    cl_report = classification_report(
        y_true= true_values['readmitted'],
        y_pred= predictions,
        zero_division=0
    )
    if print_class_report:
        print(cl_report)

    return recall, precision, f1, cl_report