from typing import Tuple, Optional, cast

import numpy as np
import pandas as pd

# leaving this here for backwards compatibility
from numerai_tools.math import (
    tie_kept_rank,
    gaussian,
    power,
    orthogonalize,
    pearson_correlation,
    tie_kept_rank__gaussianize__pow_1_5,
    tie_kept_rank__gaussianize__neutralize__variance_normalize,
    center,
    weight_normalize,
)
from numerai_tools.indexing import (
    filter_sort_index,
    filter_sort_index_many,
    filter_sort_top_bottom_concat,
    DEFAULT_MAX_FILTERED_INDEX_RATIO,
)


def correlation_contribution(
    predictions: pd.DataFrame,
    meta_model: pd.Series,
    live_targets: pd.Series,
    top_bottom: Optional[int] = None,
) -> pd.Series:
    """Calculate how much the given predictions contribute to the
    given Meta Model's correlation with the target.

    Then calculate contributive correlation by:
    1. tie-kept ranking each prediction and the meta model
    2. gaussianizing each prediction and the meta model
    3. orthogonalizing each prediction wrt the meta model
    4. dot product the orthogonalized predictions and the targets
       then normalize by the length of the target (equivalent to covariance)

    This is 100% correlated with the following formula:
    pearson_corr(
        live_targets, 0.999 * meta_model + 0.001 * predictions
    ) - pearson_corr(
        live_targets, meta_model
    )

    Arguments:
        predictions: pd.DataFrame - the predictions to evaluate
        meta_model: pd.Series - the meta model to evaluate against
        live_targets: pd.Series - the live targets to evaluate against
        top_bottom: Optional[int] - the number of top and bottom predictions to use
                                    when calculating the correlation. Results in
                                    2*top_bottom predictions.

    Returns:
        pd.Series - the resulting contributive correlation
                    scores for each column in predictions
    """
    # filter and sort preds, mm, and targets wrt each other
    meta_model, predictions = filter_sort_index(meta_model, predictions)
    live_targets, predictions = filter_sort_index(live_targets, predictions)
    live_targets, meta_model = filter_sort_index(live_targets, meta_model)

    # rank and normalize meta model and predictions so mean=0 and std=1
    p = gaussian(tie_kept_rank(predictions)).values
    m = gaussian(tie_kept_rank(meta_model.to_frame()))[meta_model.name].values

    # orthogonalize predictions wrt meta model
    neutral_preds = orthogonalize(p, cast(np.ndarray, m))

    # convert target to buckets [-2, -1, 0, 1, 2]
    if (live_targets >= 0).all() and (live_targets <= 1).all():
        live_targets = live_targets * 4
    live_targets -= live_targets.mean()

    if top_bottom is not None and top_bottom > 0:
        # filter each column to its top and bottom n predictions
        neutral_preds_df = pd.DataFrame(
            neutral_preds, columns=predictions.columns, index=predictions.index
        ).apply(lambda p: filter_sort_top_bottom_concat(p, top_bottom))
        mmc_matrix = (
            # create a dataframe for targets to match the filtered predictions
            neutral_preds_df.apply(
                lambda p: filter_sort_index(
                    p,
                    live_targets,
                    (1 - top_bottom / len(live_targets)),
                )[1]
            )
            .fillna(0)
            .T.values
            # then fill NaNs with 0 so we don't get NaNs in the dot product
            #  and mutiply target w/ neutral preds to get MMC
        ) @ neutral_preds_df.fillna(0).values
        # only the diagonal is the proper score
        mmc = np.diag(mmc_matrix) / (top_bottom * 2)
    else:
        # multiply target and neutralized predictions
        # this is equivalent to covariance b/c mean = 0
        mmc = (live_targets @ neutral_preds) / len(live_targets)
    return pd.Series(mmc, index=predictions.columns)


def numerai_corr(
    predictions: pd.DataFrame,
    targets: pd.Series,
    max_filtered_index_ratio: float = DEFAULT_MAX_FILTERED_INDEX_RATIO,
    top_bottom: Optional[int] = None,
    target_pow15: bool = True,
) -> pd.Series:
    """Calculates the canonical Numerai correlation.
    1. Re-center the target on 0
    2. filter and sort indices
    3. apply tie_kept_rank__gaussianize__pow_1_5 to the predictions
    4. raise the targets to the 1.5 power
    5. calculate the pearson correlation between the predictions and targets.

    Arguments:
        predictions: pd.DataFrame - the predictions to evaluate
        targets: pd.Series - the live targets to evaluate against
        max_filtered_index_ratio: float - the maximum ratio of indices that can be dropped
                                          when matching up the targets and predictions
        top_bottom: Optional[int] - the number of top and bottom predictions to use
                                    when calculating the correlation. Results in
                                    2*top_bottom predictions.
        target_pow15: bool - whether or not to exponentiate the target to 1.5, this
                             accentuates the tails to ensure models are performing well at
                             the extremes (where most performance comes from). Defaults to
                             true. Set to False when using returns as the "target".

    Returns:
        pd.Series - the resulting correlation scores for each column in predictions
    """
    targets = center(targets)
    targets, predictions = filter_sort_index(
        targets, predictions, max_filtered_index_ratio
    )
    predictions = tie_kept_rank__gaussianize__pow_1_5(predictions)
    if target_pow15:
        targets = power(targets.to_frame(), 1.5)[targets.name]
    scores = predictions.apply(
        lambda sub: pearson_correlation(targets, sub, top_bottom)
    )
    return scores


def feature_neutral_corr(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    targets: pd.Series,
    top_bottom: Optional[int] = None,
):
    """Calculates the canonical Numerai feature-neutral correlation.
    1. neutralize predictions relative to the features
    2. calculate the numerai_corr between the neutralized predictions and targets

    Arguments:
        predictions: pd.DataFrame - the predictions to evaluate
        features: pd.DataFrame - the features to neutralize the predictions against
        targets: pd.Series - the live targets to evaluate against
        top_bottom: Optional[int] - the number of top and bottom predictions to use
                                    when calculating the correlation. Results in
                                    2*top_bottom predictions.

    Returns:
        pd.Series - the resulting correlation scores for each column in predictions
    """
    neutral_preds = tie_kept_rank__gaussianize__neutralize__variance_normalize(
        predictions, features
    )
    return numerai_corr(neutral_preds, targets, top_bottom=top_bottom)


def max_feature_correlation(
    s: pd.Series,
    features: pd.DataFrame,
    top_bottom: Optional[int] = None,
) -> Tuple[str, float]:
    """Calculates the maximum correlation between the given series and each feature
    and returns the name of the feature and the correlation with that feature.

    Arguments:
        s: pd.Series - the series to calculate correlations against
        features: pd.DataFrame - the features to calculate correlations against
        top_bottom: Optional[int] - the number of top and bottom predictions to use
                                    when calculating the correlation. Results in
                                    2*top_bottom predictions.

    Returns:
        Tuple[str, float] - the name of the feature with the highest correlation
                            and the correlation with that feature
    """
    feature_correlations = features.apply(
        lambda f: pearson_correlation(f, s, top_bottom)
    )
    feature_correlations = feature_correlations.abs()
    max_feature = feature_correlations.idxmax()
    max_corr = feature_correlations[max_feature]
    return str(max_feature), max_corr


def generate_neutralized_weights(
    predictions: pd.DataFrame,
    neutralizers: pd.DataFrame,
    sample_weights: pd.Series,
    center_and_normalize: bool = False,
) -> pd.DataFrame:
    assert not predictions.isna().any().any(), "Predictions contain NaNs"
    assert not neutralizers.isna().any().any(), "Normalization factors contain NaNs"
    assert not sample_weights.isna().any(), "Weights contain NaNs"
    ranked_predictions = tie_kept_rank__gaussianize__pow_1_5(predictions)
    ranked_predictions, neutralizers, sample_weights = filter_sort_index_many(
        [ranked_predictions, neutralizers, sample_weights]
    )
    neutral_weights = ranked_predictions.apply(
        lambda s_prime: (
            s_prime - neutralizers @ (neutralizers.T @ (sample_weights * s_prime))
        )
        * sample_weights
    )
    if center_and_normalize:
        neutral_weights = weight_normalize(center(neutral_weights))
    return neutral_weights


def alpha(
    predictions: pd.DataFrame,
    neutralizers: pd.DataFrame,
    sample_weights: pd.Series,
    targets: pd.Series,
) -> pd.Series:
    """Calculates the "alpha" score:
        - rank, normalize, and power the signal
        - convert signal into neutralized weights
        - multiplying the weights by the targets

    Arguments:
        predictions: pd.DataFrame - the predictions to evaluate
        neutralizers: pd.DataFrame - the neutralization columns
        sample_weights: pd.Series - the universe sampling weights
        targets: pd.Series - the live targets to evaluate against
    """
    targets = center(targets)
    predictions, targets = filter_sort_index(predictions, targets)
    weights = generate_neutralized_weights(predictions, neutralizers, sample_weights)
    alpha_scores = weights.apply(lambda w: w @ targets) / len(targets)
    return alpha_scores


def meta_portfolio_contribution(
    predictions: pd.DataFrame,
    stakes: pd.Series,
    neutralizers: pd.DataFrame,
    sample_weights: pd.Series,
    targets: pd.Series,
) -> pd.Series:
    """Calculates the "meta portfolio" score:
        - rank, normalize, and power each signal
        - convert each signal into neutralized weights
        - generate the stake-weighted portfolio
        - calculate the gradient of the portfolio w.r.t. the stakes
        - multiplying the weights by the targets
    Arguments:
        predictions: pd.DataFrame - the predictions to evaluate
        stakes: pd.Series - the stakes to use as weights
        neutralizers: pd.DataFrame - the neutralization columns
        sample_weights: pd.Series - the universe sampling weights
        targets: pd.Series - the live targets to evaluate against
    """
    targets = center(targets)
    predictions, targets = filter_sort_index(predictions, targets)
    stake_weights = weight_normalize(stakes.fillna(0))
    assert np.isclose(stake_weights.sum(), 1), "Stakes must sum to 1"
    weights = generate_neutralized_weights(predictions, neutralizers, sample_weights)
    w = cast(np.ndarray, weights[stakes.index].values)
    s = cast(np.ndarray, stake_weights.values)
    t = cast(np.ndarray, targets.values)
    swp = w @ s
    swp = swp - swp.mean()
    l1_norm = np.sum(np.abs(swp))
    l1_norm_squared = np.power(l1_norm, 2)
    swp_sign = np.sign(swp)
    swp_alpha = np.dot(swp, t)
    directional_gradient = l1_norm * t - swp_sign * swp_alpha
    jacobian_vector_product = directional_gradient.reshape(-1, 1) / l1_norm_squared
    centered_jacobian = jacobian_vector_product - jacobian_vector_product.mean()
    mpc = (w.T @ centered_jacobian).squeeze()
    return pd.Series(mpc, index=stakes.index)
