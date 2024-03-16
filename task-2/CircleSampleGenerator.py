import numpy as np
from SampleGenerator import SampleGenerator


class CircleSampleGenerator(SampleGenerator):
    def __init__(self, inner_radius: float = 0.25, outer_radius: float = 0.5):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

    def generate_samples(self, num_samples=1000, error=0.1):
        num_inner_heap = num_samples // 2
        num_outer_heap = num_samples - num_inner_heap

        inner_radius_for_center = np.random.uniform(low=self.inner_radius,
                                                    high=self.inner_radius + error * self.inner_radius)
        inner_radius_for_bound = np.random.uniform(low=self.inner_radius - error * self.inner_radius,
                                                   high=self.inner_radius)

        generated_center = self._generate_samples(num_inner_heap, 0, inner_radius_for_center)
        generated_bound = self._generate_samples(num_outer_heap, inner_radius_for_bound, self.outer_radius)

        generated_center_res = np.hstack((generated_center, np.zeros((generated_center.shape[0], 1))))
        generated_bound_res = np.hstack((generated_bound, np.ones((generated_bound.shape[0], 1))))

        return np.vstack((generated_center_res, generated_bound_res))

    def _generate_samples(self, num_samples: int, inner_radius: float, outer_radius: float):
        angles = np.random.uniform(0, 2 * np.pi, num_samples)
        radii = np.random.uniform(inner_radius, outer_radius, num_samples)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack((x, y))
