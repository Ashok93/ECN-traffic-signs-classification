import matplotlib.pyplot as plt # Plotting library (https://matplotlib.org/)

def visualize_dataset(images, to_gray = True, fsize=(8,8), rows = 5, cols = 5, labels = [], predictions = []):
		fig = plt.figure(figsize=fsize)
		num_imgs = images.shape[0]
		
		
		for i in range(0, num_imgs):
			ax = fig.add_subplot(rows, cols, i + 1)
			if len(predictions) and len(labels):
				ax.set_title("Prediction: " + str(predictions[i]) + "\nTrue Label: " + str(labels[i]))
			
			# Matplot lib plots gray scale image only if dimension is like (x,y) and not (x,y,1)
			if to_gray:
				image = images[i].reshape(images[i].shape[0], images[i].shape[1])
				plt.imshow(image,  cmap='gray')
			else:
				plt.imshow(images[i])

		plt.tight_layout()
		plt.show()