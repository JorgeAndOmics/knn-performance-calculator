import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score

def determine_k_knn(dataset, k_range, distance_metric='euclidean'):
    """
    Determines the optimal value of k for K-Nearest Neighbors (KNN) algorithm.

    Parameters
    ----------
    dataset : pandas DataFrame
        The dataset to be used for determining the optimal value of k.
    k_range : list
        A list of integers representing the range of values of k to be evaluated.
    distance_metric : str, optional
        The distance metric to be used for KNN evaluation, by default 'euclidean'.

    Returns
    -------
    int
        The optimal value of k for KNN algorithm.
    """
    X = dataset.drop(columns=['target'])
    y = dataset['target']
    
    # Initialize an empty list to store the cross-validation scores
    cv_scores = []
    
    # Loop through the different values of k
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())
    
    # Determine the optimal value of k based on the highest cross-validation score
    optimal_k = k_range[np.argmax(cv_scores)]
    
    return optimal_k

def grid_search_knn(dataset, k_range, distance_metric='euclidean'):
    """
    Uses grid search to determine the optimal value of k for K-Nearest Neighbors (KNN) algorithm.

    Parameters
    ----------
    dataset : pandas DataFrame
        The dataset to be used for determining the optimal value of k.
    k_range : list
        A list of integers representing the range of values of k to be evaluated.
    distance_metric : str, optional
        The distance metric to be used for KNN evaluation, by default 'euclidean'.

    Returns
    -------
    int
        The optimal value of k for KNN algorithm.
    """
    X = dataset.drop(columns=['target'])
    y = dataset['target']
    
    knn = KNeighborsClassifier(metric=distance_metric)
    param_grid = {'n_neighbors': k_range}
    grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
    grid.fit(X, y)
    
    return grid.best_params_['n_neighbors']

def elbow_method_knn(dataset, k_range, distance_metric='euclidean'):
    """
    Uses the elbow method to determine the optimal value of k for K-Nearest Neighbors (KNN) algorithm.

    Parameters
    ----------
    dataset : pandas DataFrame
        The dataset to be used for determining the optimal value of k.
    k_range : list
        A list of integers representing the range of values of k to be evaluated.
    distance_metric : str, optional
        The distance metric to be used for KNN evaluation, by default 'euclidean'.

    Returns
    -------
    int
        The optimal value of k for KNN algorithm.
    """
    X = dataset.drop(columns=['target'])
    y = dataset['target']
    
    # Initialize an empty list to store the cross-validation scores
    cv_scores = []
    
    # Loop through the different values of k
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())
    
    # Plot the cross-validation scores to determine the "elbow"
    plt.plot(k_range, cv_scores)
    plt.xlabel('Value of k for KNN')
    plt.ylabel('Cross-Validation Accuracy')
    plt.show()
    
    # Determine the optimal value of k based on the "elbow"
    optimal_k = int(input('Enter the optimal value of k based on the elbow plot: '))
    
    return optimal_k

def plot_knn_performance(dataset, k, distance_metric='euclidean'):
        """
    Plots the performance of K-Nearest Neighbors (KNN) algorithm for a given value of k.

    Parameters
    ----------
    dataset : pandas DataFrame
        The dataset to be used for plotting the performance of KNN algorithm.
    k : int
        The value of k for which the performance of KNN algorithm is to be plotted.
    distance_metric : str, optional
        The distance metric to be used for KNN evaluation, by default 'euclidean'.

    Returns
    -------
    None
    """
    X = dataset.drop(columns=['target'])
    y = dataset['target']
    
    knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
    knn.fit(X, y)
    y_pred = knn.predict(X)
    
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    # Plot precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y, y_pred)
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="upper right")
    plt.show()
    
    # Plot accuracy score
    acc = accuracy_score(y, y_pred)
    print('Accuracy: %0.2f' % acc)

def export_results(dataset, k, distance_metric='euclidean'):
        """
    Exports the results of K-Nearest Neighbors (KNN) algorithm evaluation in a format that can be easily imported into other tools.

    Parameters
    ----------
    dataset : pandas DataFrame
        The dataset to be used for exporting the results of KNN evaluation.
    k : int
        The value of k for which the results of KNN evaluation are to be exported.
    distance_metric : str, optional
        The distance metric to be used for KNN evaluation, by default 'euclidean'.

    Returns
    -------
    None
    """
    X = dataset.drop(columns=['target'])
    y = dataset['target']
    
    knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
    knn.fit(X, y)
    y_pred = knn.predict(X)
    
    # Create a dataframe to store the results
    results = pd.DataFrame({'actual': y, 'predicted': y_pred})
    
    # Export the results in a csv file
    results.to_csv('knn_results.csv', index=False)

def batch_process_knn(datasets, k_range, distance_metric='euclidean'):
    """
    Performs a batch process for multiple datasets to determine the optimal value of k for K-Nearest Neighbors (KNN) algorithm.

    Parameters
    ----------
    datasets : list
        A list of datasets to be processed in the batch process.
    k_range : list
        A list of integers representing the range of values of k to be evaluated.
    distance_metric : str, optional
        The distance metric to be used for KNN evaluation, by default 'euclidean'.

    Returns
    -------
    dict
        A dictionary containing the optimal value of k for each dataset.
    """
    results = {}
    
    for dataset in datasets:
        optimal_k = determine_k_knn(dataset, k_range, distance_metric)
        results[dataset.name] = optimal_k
    
    return results

def search_knn(datasets, k_range, distance_metric='euclidean'):
    """
    Performs a search against a variety of datasets to determine the optimal value of k for K-Nearest Neighbors (KNN) algorithm.

    Parameters
    ----------
    datasets : list
        A list of datasets to be searched.
    k_range : list
        A list of integers representing the range of values of k to be evaluated.
    distance_metric : str, optional
        The distance metric to be used for KNN evaluation, by default 'euclidean'.

    Returns
    -------
    dict
        A dictionary containing the optimal value of k for each dataset.
    """
    results = {}
    
    for dataset in datasets:
        optimal_k = grid_search_knn(dataset, k_range, distance_metric)
        results[dataset.name] = optimal_k
    
    return results

def main():
    """
    The main function of the script.
    """
    # Load the dataset
    dataset = pd.read_csv('dataset.csv')
    
    # Set the range of values of k to be evaluated
    k_range = list(range(1, 30))
    
    # Determine the optimal value of k using the elbow method
    optimal_k = determine_k_knn(dataset, k_range)
    print('The optimal value of k using the elbow method is:', optimal_k)
    
    # Plot the performance of KNN algorithm for the optimal value of k
    plot_knn_performance(dataset, optimal_k)
    
    # Export the results of KNN evaluation
    export_results(dataset, optimal_k)
    
    # Perform a batch process for multiple datasets
    datasets = [pd.read_csv('dataset1.csv'), pd.read_csv('dataset2.csv')]
    results = batch_process_knn(datasets, k_range)
    print('The optimal value of k for each dataset in the batch process:', results)
    
    # Perform a search against a variety of datasets
    datasets = [pd.read_csv('dataset1.csv'), pd.read_csv('dataset2.csv'), pd.read_csv('dataset3.csv')]
    results = search_knn(datasets, k_range)
    print('The optimal value of k for each dataset in the search:', results)

if __name__ == '__main__':
    main()
