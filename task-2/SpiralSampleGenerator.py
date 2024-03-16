import numpy as np
from SampleGenerator import SampleGenerator


class SpiralSampleGenerator(SampleGenerator):

    def generate_samples(self, num_samples=1000, error=0.1):
        n = num_samples // 2

        def generate_samples_impl(n: int, noise: float, delta: float, label: int):
            r = np.arange(n) / n * 5
            t = 1.75 * np.arange(n) / n * 2 * np.pi + delta
            x = r * np.sin(t) + noise * np.random.uniform(-1, 1, n)
            y = r * np.cos(t) + noise * np.random.uniform(-1, 1, n)
            labels = np.full(n, label)
            return np.column_stack((x, y, labels))

        samples1 = generate_samples_impl(n, error, 0, 0)
        samples2 = generate_samples_impl(n, error, np.pi, 1)
        return np.vstack((samples1, samples2))
