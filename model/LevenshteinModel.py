import numpy as np
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm


class LevenshteinModel():

    def __init__(self):
        self.distances = None

    def distance(self, i, j, table_a, table_b):
        a = " ".join(table_a[i])
        b = " ".join(table_b[j])
        return levenshtein_distance(a, b)

    def compute_distances(self, table_a, table_b):
        len_max = max(len(table_a), len(table_b))
        self.distances = np.inf * np.ones((len_max, len_max))
        for i in tqdm(range(len(table_a)), desc="Computing distances"):
            for j in range(len(table_b)):
                self.distances[i, j] = self.distance(i, j, table_a, table_b)
        return self.distances

    def match(self, table_a, table_b):
        self.compute_distances(table_a, table_b)
        matches = []
        for i in range(len(table_a)):
            j = np.argmin(self.distances[i])
            matches.append((i, j))
        return matches

    def evaluate(self, matches, real_matches):
        accuracy = None
        precision = None
        recall = None
        f1 = None
        if real_matches is not None:
            tp = len(set(matches) & set(real_matches))
            fp = len(set(matches) - set(real_matches))
            fn = len(set(real_matches) - set(matches))
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * precision * recall / \
                (precision + recall) if precision + recall > 0 else 0
            accuracy = tp / len(real_matches) if len(real_matches) > 0 else 0
        else:
            print("No real matches provided")
        return accuracy, precision, recall, f1

    def fit(self, table_a: list, table_b: list, real_matches: list = None):
        return self.evaluate(self.match(table_a, table_b), real_matches)
