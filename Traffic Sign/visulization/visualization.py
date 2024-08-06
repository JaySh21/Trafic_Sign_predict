import matplotlib.pyplot as plt

def visualize_images(X, y, num_images=10):
    fig, axes = plt.subplots(1, num_images, figsize=(20, 20))
    for i in range(num_images):
        axes[i].imshow(X[i].squeeze(), cmap='gray')
        axes[i].set_title(f'Label: {y[i]}')
        axes[i].axis('off')
    plt.show()
