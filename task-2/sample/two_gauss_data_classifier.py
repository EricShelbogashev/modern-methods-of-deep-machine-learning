import numpy as np
from sample_generator import SampleGenerator


class TwoGaussDataClassifier(SampleGenerator):
    def generate_samples(self, num_samples=1000, error=0.1):
        variance_scale = np.interp(error, [0, 0.5], [0.5, 4])
        n = num_samples // 2

        positive_samples = np.random.normal(2, variance_scale, size=(n, 2))
        positive_labels = np.ones((n, 1))
        positive_data = np.hstack((positive_samples, positive_labels))

        negative_samples = np.random.normal(-2, variance_scale, size=(n, 2))
        negative_labels = np.zeros((n, 1))
        negative_data = np.hstack((negative_samples, negative_labels))

        return np.vstack((positive_data, negative_data))
