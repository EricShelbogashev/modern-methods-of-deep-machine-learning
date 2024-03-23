from matplotlib import pyplot as plt


class PlotGenerator:
    @staticmethod
    def plot_samples(generated, title):
        plt.figure(figsize=(6, 6))
        plt.scatter(generated[:, 0], generated[:, 1], c=generated[:, 2], s=5, cmap='Spectral')
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.colorbar(label='Class Label')
        plt.show()
