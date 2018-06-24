import matplotlib.pyplot as plt # Plotting library (https://matplotlib.org/)
import numpy as np

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

def visualize_training_data_distribution(training_labels):    
    unique, counts = np.unique(training_labels, return_counts=True)
    unique = unique.astype(int)

    fig = plt.figure()
    fig.suptitle('No. of images per class', fontsize=20)
    plt.bar(unique, counts)
    plt.xticks(np.arange(min(unique), max(unique)+1, 1.0)) #to set the x axis tick freq to 1
    plt.show()

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.jet):

    #This function prints and plots the confusion matrix.
    print('Confusion matrix, without normalization')    
    print(cm)
    fig = plt.figure()
    plt.clf()
    
    res = plt.imshow(cm, cmap=plt.cm.jet, 
                interpolation='nearest')

    width, height = cm.shape

    for x in range(width):
    	for y in range(height):
        	plt.annotate(str(cm[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center',
                    size=6
                    )
    
    cb = plt.colorbar(res)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')