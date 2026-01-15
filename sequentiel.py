import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

def kmeans_color(image, k, max_iters=200):
    pixels = image.reshape(-1, 3).astype(np.float32)
    indices = np.random.choice(len(pixels), k, replace=False)
    centers = pixels[indices]

    for _ in range(max_iters):
        distances = np.linalg.norm(pixels[:, None] - centers, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centers = []
        for i in range(k):
            cluster_pixels = pixels[labels == i]
            if len(cluster_pixels) == 0:
                new_center = pixels[np.random.randint(0, len(pixels))]
            else:
                new_center = cluster_pixels.mean(axis=0)
            new_centers.append(new_center)

        new_centers = np.array(new_centers)
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    return centers.astype(np.uint8), labels

def measure_execution_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

# Charger l'image
image = cv2.imread("image.jpg", cv2.IMREAD_COLOR)
image = cv2.GaussianBlur(image, (5, 5), 0)

k = 10
(centers, labels), execution_time = measure_execution_time(kmeans_color, image, k)
quantized_image = centers[labels].reshape(image.shape)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Image Originale')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(quantized_image, cv2.COLOR_BGR2RGB))
plt.title(f'Image Quantifiée ({k} couleurs)')
plt.axis('off')

plt.show()

print(f"\nTemps d'exécution : {execution_time:.2f} secondes pour {k} clusters.")
