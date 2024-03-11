from CircleSampleGenerator import CircleSampleGenerator
from XORSampleGenerator import XORSampleGenerator
from SpiralSampleGenerator import SpiralSampleGenerator
from TwoGaussDataClassifier import TwoGaussDataClassifier


class SampleFactory:
    def __init__(self):
        self.generators = {
            'circle': CircleSampleGenerator,
            'xor': XORSampleGenerator,
            'spiral': SpiralSampleGenerator,
            'gauss': TwoGaussDataClassifier,
        }

    def generate_samples(self, sample_type: str, num_samples=1000, error=0.1):
        if sample_type not in self.generators:
            raise ValueError(f"Unknown sample type {sample_type}")
        generator = self.generators[sample_type]()
        return generator.generate_samples(num_samples=num_samples, error=error)
