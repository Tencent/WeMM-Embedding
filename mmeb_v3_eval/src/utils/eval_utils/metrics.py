import numpy as np
import logging


logger = logging.getLogger(__name__)


class RankingMetrics:
    def __init__(self, metric_list, k_list=(1, 5, 10)):
        """
        Initialize retrieval metrics.

        Args:
            metric_list (tuple or list): Metrics to compute ("precision", "recall", "ndcg", "map", "mrr").
            k_list (tuple or list): List of K values (e.g., (1, 5, 10)) for evaluation.
        """
        self.metric_list = metric_list
        self.k_list = sorted(k_list)  # Ensure K is in ascending order

    def precision_at_k(self, prediction, true_labels, k):
        """
        Compute Precision@K for multiple true labels.
        Precision@K = (Number of relevant items in top K) / K
        """
        if not true_labels:  # No relevant items
            return 0.0
        if k == 0:
            return 0.0
        predicted_k = prediction[:k]
        relevant_hits = len(set(predicted_k).intersection(set(true_labels)))
        return relevant_hits / k

    def recall_at_k(self, prediction, true_labels, k):
        """
        Compute Recall@K for multiple true labels.
        Recall@K = (Number of relevant items in top K) / (Total number of relevant items)
        """
        if not true_labels:  # No relevant items
            return 0.0
        if k == 0 and true_labels:  # no predictions but true labels exist
            return 0.0

        predicted_k = prediction[:k]
        relevant_hits = len(set(predicted_k).intersection(set(true_labels)))
        return relevant_hits / len(true_labels)

    def _get_relevant_hits_and_predicted_k(self, prediction, true_labels, k):
        """Helper function to get common elements for calculations."""
        if k == 0:
            return 0, []
        predicted_k = prediction[:k]
        if not true_labels:  # No relevant items
            return 0, predicted_k
        relevant_hits = len(set(predicted_k).intersection(set(true_labels)))
        return relevant_hits, predicted_k

    def hit_at_k(self, prediction, true_labels, k):
        """
        Compute Hit@K (or Hit Rate@K).
        Returns 1.0 if at least one true label is found in the top K predictions, 0.0 otherwise.
        """
        if not true_labels:  # No relevant items to hit
            return 0.0
        if k == 0:
            return 0.0
        relevant_hits, _ = self._get_relevant_hits_and_predicted_k(prediction, true_labels, k)
        return 1.0 if relevant_hits > 0 else 0.0

    def f1_at_k(self, prediction, true_labels, k):
        """
        Compute F1-score@K.
        F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)
        """
        p_k = self.precision_at_k(prediction, true_labels, k)
        r_k = self.recall_at_k(prediction, true_labels, k)

        if (p_k + r_k) == 0:
            return 0.0
        return 2 * (p_k * r_k) / (p_k + r_k)

    def average_precision_at_k(self, prediction, true_labels, k):
        """
        Compute Average Precision (AP)@K for multiple true labels.
        AP is the average of precision@i for i where the i-th item is relevant.
        """
        if not true_labels:
            return 0.0  # Or 1.0 depending on definition if no relevant items. Let's use 0.0 if no true positives.
        if k == 0:
            return 0.0

        predicted_k = prediction[:k]
        score = 0.0
        num_hits = 0
        for i, p in enumerate(predicted_k):
            if p in true_labels:
                num_hits += 1
                score += num_hits / (i + 1.0)  # Precision at this new hit

        if not num_hits:  # No hits among top k
            return 0.0

        return score / min(len(true_labels), k)  # Normalize by min of total relevant or K for AP@K
        # Standard AP normalizes by total relevant items (len(true_labels))

    def mean_average_precision_at_k(self, test_cases_for_map_mrr, k_val):
        """Helper for MAP@k"""
        ap_scores = []
        for case in test_cases_for_map_mrr:
            prediction, true_labels_list = case["prediction"], case["label"]
            # Ensure true_labels_list is always a list for consistency
            if not isinstance(true_labels_list, list):
                true_labels_list = [true_labels_list]
            ap_scores.append(self.average_precision_at_k(prediction, true_labels_list, k_val))
        return np.mean(ap_scores) if ap_scores else 0.0

    def mean_reciprocal_rank_at_k(self, test_cases_for_map_mrr, k_val):
        """Helper for MRR@k"""
        rr_scores = []
        for case in test_cases_for_map_mrr:
            prediction, true_labels_list = case["prediction"], case["label"]
            if not isinstance(true_labels_list, list):
                true_labels_list = [true_labels_list]
            if not true_labels_list:
                rr_scores.append(0.0)  # Or handle as per definition, no relevant items.
                continue

            predicted_k = prediction[:k_val]
            for rank, p_item in enumerate(predicted_k):
                if p_item in true_labels_list:
                    rr_scores.append(1.0 / (rank + 1))
                    break  # Found first relevant item
            else:  # No relevant item found in top K
                rr_scores.append(0.0)
        return np.mean(rr_scores) if rr_scores else 0.0

    def ndcg_at_k(self, prediction, true_labels, k, rel_scores, form="linear"):
        """
        Compute Normalized Discounted Cumulative Gain (NDCG@K) with optional graded relevance and exponential gain.

        Args:
            prediction (list): Ranked list of predicted document IDs.
            true_labels (str or list): Relevant document(s).
            rel_scores (list or None): If provided, must be same length as label. Interpreted as graded relevance.
                                 If None, use binary relevance.
        """

        def dcg(rel_list, form):
            if form == "linear":
                return sum(((rel) / np.log2(idx + 2)) for idx, rel in enumerate(rel_list))
            elif form == "exponential":
                return sum(((2 ** rel - 1) / np.log2(idx + 2)) for idx, rel in enumerate(rel_list))

        if rel_scores is None:
            # Binary relevance
            if isinstance(true_labels, list):
                label_set = set(true_labels)
            else:
                label_set = {true_labels}
            relevance = [1 if item in label_set else 0 for item in prediction[:k]]
            ideal_relevance = [1] * min(len(label_set), k)
        else:
            # Graded relevance
            if not isinstance(true_labels, list) or not isinstance(
                    rel_scores, list) or len(true_labels) != len(rel_scores):
                raise ValueError("If rels is provided, label must be a list of the same length.")
            label_rels = dict(zip(true_labels, rel_scores))
            relevance = [label_rels.get(item, 0) for item in prediction[:k]]
            ideal_relevance = sorted(label_rels.values(), reverse=True)[:k]
        dcg_score = dcg(relevance, form)
        idcg_score = dcg(ideal_relevance, form) if ideal_relevance else 1  # Avoid division by zero

        return dcg_score / idcg_score if idcg_score > 0 else 0.0

    def evaluate(self, test_cases):
        """
        Evaluates predictions against true labels for a list of test cases.
        Handles multi-label by ensuring `label` is a list of true positives.

        Args:
            test_cases (list of dict): Each dict should have "prediction" (list)
                                       and "label" (list of true positive labels).
        """
        metric_results_accumulators = {
            (f"ndcg_{v}" if metric == "ndcg" else metric): {k: [] for k in self.k_list}
            for metric in self.metric_list
            for v in (["linear", "exponential"] if metric == "ndcg" else [None])
        }

        processed_test_cases_for_map_mrr = []

        for case_idx, case in enumerate(test_cases):
            prediction = case["prediction"]
            true_labels = case["label"]
            if not isinstance(true_labels, (list, set)):  # Allow set for true_labels
                true_labels = [true_labels]
            # Ensure unique true labels while preserving order: rel_scores (if any)
            # is zipped positionally with the original label order, and set()
            # would scramble that alignment for graded relevance.
            true_labels = list(dict.fromkeys(true_labels))

            # For MAP and MRR, we often want the original list of true_labels,
            # even if empty, for consistent calculation across cases.
            if "map" in self.metric_list or "mrr" in self.metric_list:
                processed_test_cases_for_map_mrr.append(
                    {"prediction": prediction, "label": true_labels, "id": case_idx}
                )

            for k in self.k_list:
                if "precision" in self.metric_list:
                    metric_results_accumulators["precision"][k].append(self.precision_at_k(prediction, true_labels, k))
                if "recall" in self.metric_list:
                    metric_results_accumulators["recall"][k].append(self.recall_at_k(prediction, true_labels, k))
                if "hit" in self.metric_list:
                    metric_results_accumulators["hit"][k].append(self.hit_at_k(prediction, true_labels, k))
                if "f1" in self.metric_list:
                    metric_results_accumulators["f1"][k].append(self.f1_at_k(prediction, true_labels, k))
                if "ndcg" in self.metric_list:
                    rel_scores = case["rel_scores"]
                    metric_results_accumulators["ndcg_linear"][k].append(
                        self.ndcg_at_k(prediction, true_labels, k, rel_scores, form="linear"))
                    metric_results_accumulators["ndcg_exponential"][k].append(
                        self.ndcg_at_k(prediction, true_labels, k, rel_scores, form="exponential"))

        # Calculate final mean for most metrics
        score_dict = {}
        for metric, k_values_dict in metric_results_accumulators.items():
            if metric not in ["map", "mrr"]:  # MAP/MRR calculated differently
                for k, values in k_values_dict.items():
                    score_dict[f"{metric}@{k}"] = float(np.mean(values)) if values else 0.0

        # Calculate MAP and MRR across all test cases for each k
        if "map" in self.metric_list:
            for k_val in self.k_list:
                ap_scores = [
                    self.average_precision_at_k(case["prediction"], case["label"], k_val)
                    for case in processed_test_cases_for_map_mrr
                ]
                score_dict[f"map@{k_val}"] = float(np.mean(ap_scores)) if ap_scores else 0.0

        if "mrr" in self.metric_list:
            rr_scores = []
            for case in processed_test_cases_for_map_mrr:
                reciprocal_rank = 0.0
                if case["label"]:  # Only if there are true labels
                    for rank, p_item in enumerate(case["prediction"]):
                        if p_item in case["label"]:
                            reciprocal_rank = 1.0 / (rank + 1)
                            break
                rr_scores.append(reciprocal_rank)
            score_dict["mrr"] = float(np.mean(rr_scores)) if rr_scores else 0.0

        return score_dict
