"""
The :mod:`scikitplot.decomposition` module includes plots built specifically
for scikit-learn estimators that are used for dimensionality reduction
e.g. PCA. You can use your own estimators, but these plots assume specific
properties shared by scikit-learn estimators. The specific requirements are
documented per function.
"""
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
x = np.array([0,.2,.4,.6,.8,1])

def plot_pca_component_variance(clf, title='PCA Component Explained Variances',
                                target_explained_variance=0.75, ax=None,
                                figsize=(10,6), title_fontsize=18,
                                text_fontsize=14, bbox_to_anchor = (1.4,.5),
                                **kwargs):
    """Plots PCA components' explained variance ratios. (new in v0.2.2)

    Args:
        clf: PCA instance that has the ``explained_variance_ratio_`` attribute.

        title (string, optional): Title of the generated plot. Defaults to
            "PCA Component Explained Variances"

        target_explained_variance (float, optional): Looks for the minimum
            number of principal components that satisfies this value and
            emphasizes it on the plot. Defaults to 0.75

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
        >>> pca = PCA(random_state=1)
        >>> pca.fit(X)
        >>> skplt.decomposition.plot_pca_component_variance(pca)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_pca_component_variance.png
           :align: center
           :alt: PCA Component variances
    """
    if not hasattr(clf, 'explained_variance_ratio_'):
        raise TypeError('"clf" does not have explained_variance_ratio_ '
                        'attribute. Has the PCA been fitted?')

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize, pad = 20,
                fontfamily = 'serif', color = '#525252')

    cumulative_sum_ratios = np.cumsum(clf.explained_variance_ratio_)

    # Magic code for figuring out closest value to target_explained_variance
    idx = np.searchsorted(cumulative_sum_ratios, target_explained_variance)

    ax.plot(range(len(clf.explained_variance_ratio_) + 1),
            np.concatenate(([0], np.cumsum(clf.explained_variance_ratio_))),
            '*-')
    ax.grid(True)
    ax.set_xlabel('First n principal components', fontsize=text_fontsize,
                 fontfamily = 'serif', color = '#525252', labelpad = 15)
    
    ax.set_ylabel('Explained variance ratio of first n components',
                  fontsize=text_fontsize, fontfamily = 'serif', 
                  color = '#525252', labelpad = 15)
    
    ax.set_ylim([-0.02, 1.02])
    
    if idx < len(cumulative_sum_ratios):
        ax.plot(idx+1, cumulative_sum_ratios[idx], 'ro',
                label='{0:0.3f} Explained variance \nratio for '
                'first {1} components'.format(cumulative_sum_ratios[idx],
                                              idx+1),
                markersize=4, markeredgewidth=4)
        ax.axhline(cumulative_sum_ratios[idx],
                   linestyle=':', lw=3, color='black')
    
    ax.tick_params(colors='#525252', labelsize=text_fontsize -2)
    
    leg = ax.legend(loc='lower right', fontsize=text_fontsize - 2, 
                    edgecolor = 'white',  bbox_to_anchor = bbox_to_anchor, 
                    **kwargs)
    
    for j in range(len(leg.texts)):
        leg.texts[j].set_color('#4f4f4f')
        leg.texts[j].set_family('serif')
    
    sns.despine(trim = True, offset = 5)
    
    sns.despine(offset = 5)
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['bottom'].set_linewidth(2)
    
    ax.spines['left'].set_color('#cccccc')
    ax.spines['left'].set_linewidth(2)
    
    ax.grid(color = '#cccccc', alpha = 0.3)
    
    ax.set_xticklabels(ax.get_xticks(), color = '#525252',fontfamily = 'serif')
    ax.set_yticklabels(x, color = '#525252',fontfamily = 'serif')
    
    return ax


def plot_pca_2d_projection(clf, X, y, title='PCA 2-D Projection',
                           biplot=False, feature_labels=None,
                           ax=None, figsize=(7,7), cmap='Spectral',
                           title_fontsize=18, text_fontsize=14):
    """Plots the 2-dimensional projection of PCA on a given dataset.

    Args:
        clf: Fitted PCA instance that can ``transform`` given data set into 2
            dimensions.

        X (array-like, shape (n_samples, n_features)):
            Feature set to project, where n_samples is the number of samples
            and n_features is the number of features.

        y (array-like, shape (n_samples) or (n_samples, n_features)):
            Target relative to X for labeling.

        title (string, optional): Title of the generated plot. Defaults to
            "PCA 2-D Projection"

        biplot (bool, optional): If True, the function will generate and plot
        	biplots. If false, the biplots are not generated.

        feature_labels (array-like, shape (n_classes), optional): List of labels
        	that represent each feature of X. Its index position must also be
        	relative to the features. If ``None`` is given, then labels will be
        	automatically generated for each feature.
        	e.g. "variable1", "variable2", "variable3" ...

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html

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
        >>> pca = PCA(random_state=1)
        >>> pca.fit(X)
        >>> skplt.decomposition.plot_pca_2d_projection(pca, X, y)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_pca_2d_projection.png
           :align: center
           :alt: PCA 2D Projection
    """
    transformed_X = clf.transform(X)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)
    classes = np.unique(np.array(y))

    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(classes)))

    for label, color in zip(classes, colors):
        ax.scatter(transformed_X[y == label, 0], transformed_X[y == label, 1],
                   alpha=0.8, lw=2, label=label, color=color)

    if biplot:
        xs = transformed_X[:, 0]
        ys = transformed_X[:, 1]
        vectors = np.transpose(clf.components_[:2, :])
        vectors_scaled = vectors * [xs.max(), ys.max()]
        for i in range(vectors.shape[0]):
            ax.annotate("", xy=(vectors_scaled[i, 0], vectors_scaled[i, 1]),
                        xycoords='data', xytext=(0, 0), textcoords='data',
                        arrowprops={'arrowstyle': '-|>', 'ec': 'r'})

            ax.text(vectors_scaled[i, 0] * 1.05, vectors_scaled[i, 1] * 1.05,
                    feature_labels[i] if feature_labels else "Variable" + str(i),
                    color='b', fontsize=text_fontsize)

    ax.legend(loc='best', shadow=False, scatterpoints=1,
              fontsize=text_fontsize)
    ax.set_xlabel('First Principal Component', fontsize=text_fontsize)
    ax.set_ylabel('Second Principal Component', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)

    return ax


def plot_labelled_scatter(X, y, class_labels):
    
    num_labels = len(class_labels)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    marker_array = ['o', '^', '*']
    color_array = ['#FFFF00', '#00AAFF', '#000000', '#FF00AA']
    cmap_bold = ListedColormap(color_array)
    bnorm = BoundaryNorm(numpy.arange(0, num_labels + 1, 1), ncolors=num_labels)
    plt.figure()

    plt.scatter(X[:, 0], X[:, 1], s=65, c=y, cmap=cmap_bold, norm = bnorm, alpha = 0.40, edgecolor='black', lw = 1)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    h = []
    for c in range(0, num_labels):
        h.append(mpatches.Patch(color=color_array[c], label=class_labels[c]))
    plt.legend(handles=h)

    plt.show()