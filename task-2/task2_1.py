import sys
sys.path.append('sample')
from sample.plot_generator import PlotGenerator
from sample.sample_factory import SampleFactory

if __name__ == '__main__':
    # circle, xor, spiral, gauss
    sample_type = 'circle'
    generated = SampleFactory().generate_samples(sample_type=sample_type, num_samples=1000, error=0.5)
    PlotGenerator.plot_samples(generated, sample_type)
