from SampleFactory import SampleFactory
from PlotGenerator import PlotGenerator

if __name__ == '__main__':
    # circle, xor, spiral, gauss
    sample_type = 'spiral'
    generated = SampleFactory().generate_samples(sample_type=sample_type, num_samples=1000, error=0.5)
    PlotGenerator.plot_samples(generated, sample_type)
