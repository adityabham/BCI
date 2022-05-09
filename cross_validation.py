import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from show_chanWeights import show_chanWeights


def mean_roc_data_gen(tpr_data, fpr_data):
    adjusted_data = []
    for idx in range(0, len(tpr_data)):
        adj_trp = np.interp(np.linspace(0, 1, 1000), fpr_data[idx], tpr_data[idx])
        adj_trp[0] = 0.0
        adjusted_data.append(adj_trp)

    return np.linspace(0, 1, 1000), np.mean(adjusted_data, axis=0)


def weights_util(weights):
    dom_chan_ind = (-np.absolute(weights)).argsort()[:5]
    dom_weights = weights[dom_chan_ind]
    dom_chan = dom_chan_ind + 1
    print("Dominant Channels:")
    print(dom_chan)
    print("Corresponding Weights:")
    print(dom_weights)

    show_chanWeights(np.absolute(weights))
    plt.figure(figsize=(9, 6))
    plt.plot(list(range(1, 205)), weights, color='black', linewidth=1.5)
    plt.scatter(dom_chan, dom_weights)
    plt.title("Weight Per Channel Visualization")
    plt.xlabel("Channel")
    plt.ylabel("Weight")
    plt.show()


def CV(X, y, params):
    cv_first_level = StratifiedKFold(n_splits=6, shuffle=True, random_state=1)

    perf_scores = []
    fold_tpr_data = []
    fold_fpr_data = []
    first_fold_weights = []
    plt.figure(figsize=(6, 6))
    for i, (train, test) in enumerate(cv_first_level.split(X, y)):
        X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]

        cv_second_level = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        linear_svm = SVC(kernel='linear', random_state=1, probability=True)
        grid_search = GridSearchCV(linear_svm, param_grid=dict(C=params), n_jobs=-1,
                                   cv=cv_second_level, refit=True)
        grid_search.fit(X_train, y_train)
        optimal_model = grid_search.best_estimator_
        y_pred = optimal_model.predict(X_test)
        perf_score = accuracy_score(y_test, y_pred)
        print("Fold " + str(i + 1) + " Accuracy Score:")
        print(perf_score)
        print(grid_search.best_params_)
        print("----------------")
        perf_scores.append(perf_score)
        if i == 0:
            first_fold_weights.append(optimal_model.coef_[0])

        fpr, tpr, _ = roc_curve(y_test, optimal_model.predict_proba(X_test)[:, 1])
        fold_fpr_data.append(fpr)
        fold_tpr_data.append(tpr)
        plt.plot(fpr, tpr, linewidth=0.5, label='Iteration: ' + str(i + 1))
    mean_roc_x, mean_roc_y = mean_roc_data_gen(fold_tpr_data, fold_fpr_data)
    print("AUC:" + str(auc(mean_roc_x, mean_roc_y)))

    print("Total Cross-Validated Accuracy Score:")
    print(sum(perf_scores) / len(perf_scores))
    print("----------------")

    plt.plot([0, 1], [0, 1], '--')
    plt.plot(mean_roc_x, mean_roc_y, linewidth=1.5, label='Mean ROC')
    plt.title("ROC CURVE (Simulation 1)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    weights_util(first_fold_weights[0])
