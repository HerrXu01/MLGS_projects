import numpy as np


def classify_review(hmm_1, hmm_5, p, sentence_in):
    """Given the trained models `hmm_1` and `hmm_2` and frequency of
       1-star reviews, classifies `sentence_in`

    Parameters
    ----------
    hmm_1 : HMM_TxtGenerator
        The trained model on 1-star reviews.
    hmm_5 : HMM_TxtGenerator
        The trained model on 5-star reviews.
    p: a scalar in [0,1]
        frequency of 1-star reviews, (#1star)/(#1star + #5star)

    Returns
    -------
    c : int in {1,5}
        c=1 means sentence_in is classified as 1.
        similarly c=5 means sentence_in is classified as 5.
        If both sentences are equally likely, you can return either 1 or 5.
    """

    ### YOUR CODE HERE ###
    prob_1 = np.log(p) + hmm_1.loglik_sentence(sentence_in)
    prob_5 = np.log(1-p) + hmm_5.loglik_sentence(sentence_in)
    if prob_1 > prob_5:
        return 1
    else:
        return 5
