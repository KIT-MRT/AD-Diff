# file modified from original Apache-2.0 licensed code from https://github.com/Understanding-Visual-Datasets/VisDiff
# see LICENSE and NOTICE files in the root directory for details

from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from numpy import random
import math

import wandb
from serve.utils_clip import get_embeddings


def plot_distributions(similarity_A_C, similarity_B_C, hypothesis="", expect_singular=False):
    """
    Plots the distributions of cos sim to hypothesis for each group.
    """
    # Convert arrays to 1D if they're 2D
    similarity_A_C = np.array(similarity_A_C).ravel()
    similarity_B_C = np.array(similarity_B_C).ravel()

    # Create a combined list of all scores and a list of labels to indicate group membership
    all_scores = list(similarity_A_C) + list(similarity_B_C)
    labels = ["Group A"] * len(similarity_A_C) + ["Group B"] * len(similarity_B_C)

    # Create a DataFrame for seaborn plotting
    df = pd.DataFrame({"Group": labels, "Similarity to C": all_scores})

    # Set up the figure with 3 subplots
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

    # Histogram
    ax[0].hist(similarity_A_C, bins=30, alpha=0.5, label="Group A", density=True)
    ax[0].hist(similarity_B_C, bins=30, alpha=0.5, label="Group B", density=True)
    ax[0].set_title(f"Histogram of Cosine Similarities to \n{hypothesis}")
    ax[0].set_ylabel("Density")
    ax[0].legend()

    # KDE plot
    sns.kdeplot(similarity_A_C, fill=True, ax=ax[1], label="Group A", warn_singular=not expect_singular)
    sns.kdeplot(similarity_B_C, fill=True, ax=ax[1], label="Group B", warn_singular=not expect_singular)
    ax[1].set_title(
        f"Kernel Density Estimation of Cosine Similarities to \n{hypothesis}"
    )
    ax[1].set_ylabel("Density")

    # Boxplot
    sns.boxplot(x="Group", y="Similarity to C", data=df, ax=ax[2])
    ax[2].set_title(f"Boxplot of Cosine Similarities to \n{hypothesis}")

    # Adjust layout
    plt.tight_layout()
    return fig


def classify(similarity_A_C, similarity_B_C, threshold=0.3):
    """
    Given two arrays of cos sim scores, classify each item of each group as containing concept C or not.
    Return P(hyp in A) - P(hyp in B)
    """
    similarity_A_C = np.array(similarity_A_C)
    similarity_B_C = np.array(similarity_B_C)
    # print(
    #     f"avg(cos sim A, cos sim B) = {[np.mean(similarity_A_C), np.mean(similarity_B_C)]} \t Max(cos sim A, cos sim B) = {[np.max(similarity_A_C), np.max(similarity_B_C)]}"
    # )
    percent_correct_a = sum(similarity_A_C > threshold) / len(similarity_A_C)
    percent_correct_b = sum(similarity_B_C > threshold) / len(similarity_B_C)
    # print(f"Percent correct A, B {[percent_correct_a, percent_correct_b]}")
    return percent_correct_a - percent_correct_b


def compute_auroc(similarity_A_C, similarity_B_C):
    similarity_A_C = np.array(similarity_A_C)
    similarity_B_C = np.array(similarity_B_C)

    # Create labels based on the sizes of the input arrays
    labels_A = [1] * similarity_A_C.shape[0]
    labels_B = [0] * similarity_B_C.shape[0]

    # Concatenate scores and labels using numpy's concatenate
    all_scores = np.concatenate([similarity_A_C, similarity_B_C], axis=0).ravel()
    all_labels = labels_A + labels_B

    # Compute AUROC
    auroc = roc_auc_score(all_labels, all_scores)
    return auroc


def t_test(d_A, d_B):
    d_A = np.array(d_A)
    d_B = np.array(d_B)

    # Assuming you've already defined your similarity scores d_A and d_B
    t_stat, p_value = ttest_ind(d_A, d_B, equal_var=False)

    # Decision
    alpha = 0.05
    if p_value < alpha:
        # print("** Reject the null hypothesis - there's a significant difference between the groups. **")
        return True, p_value
    else:
        # print("Fail to reject the null hypothesis - there's no significant difference between the groups.")
        return False, p_value


class Ranker:
    def __init__(self, args: Dict, rng: random.Generator, expect_singular: bool = False):
        self.args = args
        self.rng = rng
        self.expect_singular = expect_singular

    def _hypothesis_str(self, hypothesis: str|dict) -> str:
        return hypothesis if isinstance(hypothesis, str) else hypothesis["hypothesis"]

    def _subsample_precalculated_scores(self, hypotheses: List[str]|List[dict], dataset: List[dict], indices: List[int], scores_key: str) -> List[str]|List[dict]:
        if isinstance(hypotheses[0], dict) and scores_key in hypotheses[0]:
            for h in hypotheses:
                h[scores_key] = [h[scores_key][i] for i in indices]
        return hypotheses

    def score_hypothesis(self, hypothesis: str|dict, dataset: List[dict], set1: bool) -> List[float]:
        raise NotImplementedError

    def rerank_hypotheses(
        self, hypotheses: List[str]|List[dict], dataset1: List[dict], dataset2: List[dict]
    ) -> List[dict]:
        if len(dataset1) > self.args.get("max_num_samples", math.inf):
            indices = self.rng.choice(len(dataset1), self.args["max_num_samples"], replace=False)
            dataset1 = [dataset1[i] for i in indices]
            hypotheses = self._subsample_precalculated_scores(hypotheses, dataset1, indices, "scores1")
        if len(dataset2) > self.args.get("max_num_samples", math.inf):
            indices = self.rng.choice(len(dataset2), self.args["max_num_samples"], replace=False)
            dataset2 = [dataset2[i] for i in indices]
            hypotheses = self._subsample_precalculated_scores(hypotheses, dataset2, indices, "scores2")

        scored_hypotheses = []
        for hypothesis in tqdm(hypotheses):
            scores1 = self.score_hypothesis(hypothesis, dataset1, set1=True)
            scores2 = self.score_hypothesis(hypothesis, dataset2, set1=False)

            metrics = self.compute_metrics(scores1, scores2, self._hypothesis_str(hypothesis))
            scored_hypotheses.append(metrics)
        if self.args.get("sort_key", None):
            scored_hypotheses = sorted(
                scored_hypotheses, key=lambda x: x[self.args["sort_key"]], reverse=self.args["sort_descending"]
            )
        return scored_hypotheses

    def compute_metrics(
        self, scores1: List[float], scores2: List[float], hypothesis: str
    ) -> dict:
        metrics = {}
        metrics["hypothesis"] = hypothesis
        metrics["score1"] = np.mean(scores1)
        metrics["score2"] = np.mean(scores2)
        metrics["diff"] = metrics["score1"] - metrics["score2"]
        metrics["t_stat"], metrics["p_value"] = t_test(scores1, scores2)
        metrics["auroc"] = compute_auroc(scores1, scores2)
        metrics["correct_delta"] = classify(
            scores1, scores2, threshold=self.args.get("classify_threshold", 0.5)
        )
        metrics["distribution"] = wandb.Image(
            plot_distributions(scores1, scores2, hypothesis=hypothesis, expect_singular=self.expect_singular)
        )
        return metrics


class CLIPRanker(Ranker):
    def __init__(self, args: Dict, rng: random.Generator):
        super().__init__(args, rng=rng)
        self.clip_hostname = args.get("clip_hostname", "localhost")
        print(f"Initialized ranker with CLIP hostname: {self.clip_hostname}")

    def score_hypothesis(self, hypothesis: str|dict, dataset: List[dict], set1: bool) -> List[float]:
        model = self.args["model"]
        image_features = get_embeddings(
            [item["path"] for item in dataset], model, "image", self.clip_hostname
        )
        text_features = get_embeddings([self._hypothesis_str(hypothesis)], model, "text", self.clip_hostname)
        similarity = (image_features @ text_features.T) / (
            np.linalg.norm(image_features, axis=1, keepdims=True)
            * np.linalg.norm(text_features, axis=1, keepdims=True).T
            + 1e-10
        )
        scores = similarity.squeeze(1).tolist()
        return scores

class NullRanker(Ranker):
    """
    NullRanker does not actually rank hypotheses, it reuses scores precalculated by the proposer if they exist, and if no precalculated scores exist, it returns dummy scores that will lead to a tie between all hypotheses.

    Use NullRanker if hypotheses are already ranked by the proposer, or should be ranked based on scores precalculated by the proposer.
    """
    def __init__(self, args: Dict, rng: random.Generator):
        super().__init__(args, rng=rng, expect_singular=True)

    def score_hypothesis(self, hypothesis: str|dict, dataset: List[dict], set1: bool) -> List[float]:
        # if the hypothesis already has precalculated scores, use those (this is useful for proposers that calculate scores on the subsample and want to reuse those scores for reranking on the full sample)
        try:
            # one score per item in the dataset, use for distributional metrics to rank hypotheses
            if set1:
                scores = hypothesis["score1"]
                assert len(scores) == len(dataset), f"Length of scores1 {len(scores)} does not match length of dataset {len(dataset)}"
            else:
                scores = hypothesis["score2"]
                assert len(scores) == len(dataset), f"Length of scores2 {len(scores)} does not match length of dataset {len(dataset)}"
        except (KeyError, TypeError):
            try:
                # one score for the whole dataset: proposer already defined final score for the hypothesis
                score = hypothesis["score"]
            except (KeyError, TypeError):
                # no precalculated scores, return dummy scores that will lead to a tie between all hypotheses
                score = 0.0
            scores = [score] * len(dataset)
        return scores
