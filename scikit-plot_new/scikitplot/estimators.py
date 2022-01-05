"""
The :mod:`scikitplot.estimators` module includes plots built specifically for
scikit-learn estimator (classifier/regressor) instances e.g. Random Forest.
You can use your own estimators, but these plots assume specific properties
shared by scikit-learn estimators. The specific requirements are documented per
function.
"""
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import learning_curve, validation_curve
import seaborn as sns


def plot_feature_importances(clf, title='Feature Importance',
                             feature_names=None, ax=None, error = False,
                             figsize=(8,6), title_fontsize=18,
                             text_fontsize=14):
    """Generates a plot of a classifier's feature importances.

    Args:
        clf: Classifier instance that has a ``feature_importances_`` attribute,
            e.g. :class:`sklearn.ensemble.RandomForestClassifier` or
            :class:`xgboost.XGBClassifier`.

        title (string, optional): Title of the generated plot. Defaults to
            "Feature importances".

        feature_names (None, :obj:`list` of string, optional): Determines the
            feature names used to plot the feature importances. If None,
            feature names will be numbered.

        order ('ascending', 'descending', or None, optional): Determines the
            order in which the feature importances are plotted. Defaults to
            'descending'.

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot as skplt
        >>> rf = RandomForestClassifier()
        >>> rf.fit(X, y)
        >>> skplt.estimators.plot_feature_importances(
        ...     rf, feature_names=['petal length', 'petal width',
        ...                        'sepal length', 'sepal width'])
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_feature_importances.png
           :align: center
           :alt: Feature Importances
    """
   
    if not hasattr(clf, 'feature_importances_'):
        raise TypeError('"feature_importances_" attribute not in classifier. '
                        'Cannot plot feature importances.')

    importances = clf.feature_importances_
    
    if error:
        if hasattr(clf, 'estimators_')\
                and isinstance(clf.estimators_, list)\
                and hasattr(clf.estimators_[0], 'feature_importances_'):
            std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                         axis=0)
        else:
            std = None
    else:
        std = None
        
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    c_features = len(feature_names)
    if feature_names is None:
        feature_names = range(c_features)
    else:
        feature_names = np.array(feature_names)
        
    c_features = len(feature_names)
    
    sns.despine(offset = 5)
    ax.set_title(title, fontsize=title_fontsize,pad = 25,
                fontfamily = 'serif', color = '#525252')

    if std is not None:
        ax.barh(range(c_features), importances,
               xerr=std, align = 'center', color = 'steelblue')
    else:
        ax.barh(range(c_features), importances,
                  color = 'steelblue')
    
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['bottom'].set_linewidth(2)
    
    ax.spines['left'].set_color('#cccccc')
    ax.spines['left'].set_linewidth(2)
    
    ax.tick_params(colors='#525252', labelsize=text_fontsize -2)
    y = np.arange(c_features)
    
    ax.set_ylim([-0.5,y.shape[0] -0.5 ])
    
    ax.set_yticks(np.arange(c_features), feature_names,
                  color = '#525252', fontfamily = 'serif',va = 'center')
    
    return ax


def plot_learning_curve(clf, X, y, title='Learning Curve', cv=5,
                        shuffle=False, random_state=None,
                        train_sizes=None, n_jobs=1, scoring=None,
                        ax=None, figsize=(9,5), title_fontsize=18,
                        text_fontsize=14, bbox_to_anchor = (1.02,.4), **kwargs):
    """Generates a plot of the train and test learning curves for a classifier.

    Args:
        clf: Classifier instance that implements ``fit`` and ``predict``
            methods.

        X (array-like, shape (n_samples, n_features)):
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y (array-like, shape (n_samples) or (n_samples, n_features)):
            Target relative to X for classification or regression;
            None for unsupervised learning.

        title (string, optional): Title of the generated plot. Defaults to
            "Learning Curve"

        cv (int, cross-validation generator, iterable, optional): Determines
            the cross-validation strategy to be used for splitting.

            Possible inputs for cv are:
              - None, to use the default 3-fold cross-validation,
              - integer, to specify the number of folds.
              - An object to be used as a cross-validation generator.
              - An iterable yielding train/test splits.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is
            used.

        shuffle (bool, optional): Used when do_cv is set to True. Determines
            whether to shuffle the training data before splitting using
            cross-validation. Default set to True.

        random_state (int :class:`RandomState`): Pseudo-random number generator
            state used for random sampling.

        train_sizes (iterable, optional): Determines the training sizes used to
            plot the learning curve. If None, ``np.linspace(.1, 1.0, 5)`` is
            used.

        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to
            1.

        scoring (string, callable or None, optional): default: None
            A string (see scikit-learn model evaluation documentation) or a
            scorerbcallable object / function with signature
            scorer(estimator, X, y).

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".
            
        bbox_to_anchor (tuple, optional) its the bbox of legend
        
        **kwargs arguments to legend 

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot as skplt
        >>> rf = RandomForestClassifier()
        >>> skplt.estimators.plot_learning_curve(rf, X, y)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_learning_curve.png
           :align: center
           :alt: Learning Curve
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if train_sizes is None:
        train_sizes = np.linspace(.1, 1.0, 10)

    ax.set_title(title, fontsize=title_fontsize, pad = 25,
                fontfamily = 'serif', color = '#525252')
    
    ax.set_xlabel("Training examples", fontsize=text_fontsize,
                 color = '#525252',fontfamily = 'serif', labelpad = 15)
    ax.set_ylabel("Score", fontsize=text_fontsize,
                 color = '#525252',fontfamily = 'serif', labelpad = 15)
    
    train_sizes_, train_scores, test_scores = learning_curve(
        clf, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring=scoring, shuffle=shuffle, random_state=random_state)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")
    
    ax.tick_params(colors='#525252', labelsize=text_fontsize -2)
    
    
    sns.despine(offset = 5)
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['bottom'].set_linewidth(2)
    
    ax.spines['left'].set_color('#cccccc')
    ax.spines['left'].set_linewidth(2)
    
    leg = ax.legend(loc='lower left', fontsize=text_fontsize - 2, 
                    edgecolor = 'white',  bbox_to_anchor = bbox_to_anchor, 
                    **kwargs)
    
    
    ax.set_xticks(train_sizes, color = '#525252',fontfamily = 'serif')
    #ax.grid(color = '#cccccc', axis = 'both', alpha = 0.3)
    for j in range(len(leg.texts)):
        leg.texts[j].set_color('#4f4f4f')
        leg.texts[j].set_family('serif')


    return ax

def plot_validation_curve(clf, X, y, param_name, param_range,
                          scoring = 'accuracy', n_jobs = 1, title = 'Validation Curve',
                          cv=5, ax = None, figsize = (10,5), title_fontsize=18,
                          text_fontsize=14, bbox_to_anchor = (1.35, .65), **kwargs):
    """Generates a plot of the train and test learning curves for a classifier.

    Args:
        clf: Classifier instance that implements ``fit`` and ``predict``
            methods.

        X (array-like, shape (n_samples, n_features)):
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y (array-like, shape (n_samples) or (n_samples, n_features)):
            Target relative to X for classification or regression;
            None for unsupervised learning.
            
        param_name : str
        Name of the parameter that will be varied.

        param_range: array-like of shape (n_values,)
        The values of the parameter that will be evaluated.
        
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to
            1.

        scoring (string, callable or None, optional): default: None
            A string (see scikit-learn model evaluation documentation) or a
            scorerbcallable object / function with signature
            scorer(estimator, X, y). 
            
        cv (int, cross-validation generator, iterable, optional): Determines
            the cross-validation strategy to be used for splitting.

            Possible inputs for cv are:
              - None, to use the default 3-fold cross-validation,
              - integer, to specify the number of folds.
              - An object to be used as a cross-validation generator.
              - An iterable yielding train/test splits.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is
            used.
            
        title (string, optional): Title of the generated plot. Defaults to
            "Validation Curve"

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".
            
        bbox_to_anchor (tuple, optional) its the bbox of legend
        
        **kwargs arguments to legend             
            
    """
    train_scores, test_scores = validation_curve(clf, X, y, param_name=param_name,
                                                 param_range=param_range, scoring=scoring,
                                                 n_jobs=n_jobs, cv = cv)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
    ax.set_title(title, fontsize=title_fontsize, pad = 25,
                fontfamily = 'serif', color = '#525252')
    
    ax.set_xlabel("Parameter Range", fontsize=text_fontsize,
                 color = '#525252',fontfamily = 'serif', labelpad = 15)
    ax.set_ylabel("Score", fontsize=text_fontsize,
                 color = '#525252',fontfamily = 'serif', labelpad = 15)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    ax.semilogx(param_range, train_scores_mean, 'o-',
                 label="Training score", color="red", lw=3)
    
    ax.fill_between(param_range, train_scores_mean - train_scores_std,
                                  train_scores_mean + train_scores_std,
                     alpha=0.2, color="red",lw=3)
    
    ax.semilogx(param_range, test_scores_mean, 'o-',
                 label="Cross-validation score", color="green", lw=3)
    
    ax.fill_between(param_range, test_scores_mean - test_scores_std,
                                  test_scores_mean + test_scores_std,
                     alpha=0.2, color="green", lw=3)
    
    sns.despine(offset = 5)
    
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['bottom'].set_linewidth(2)
    
    ax.spines['left'].set_color('#cccccc')
    ax.spines['left'].set_linewidth(2)
    
    leg = ax.legend(loc="best", fontsize=text_fontsize - 2, 
                    edgecolor = 'white',  bbox_to_anchor = bbox_to_anchor, 
                    **kwargs)
    
    ax.tick_params(colors='#525252', labelsize=text_fontsize -2)
    
    for j in range(len(leg.texts)):
        leg.texts[j].set_color('#4f4f4f')
        leg.texts[j].set_family('serif')
        
    return ax




