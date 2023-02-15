import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# This is similar to the program in 2_2_A.py but uses the ERM classification rule with the following loss matrices which care 10 or 100 times more about not making mistakes when L = 3

A10 = [[0, 1, 1], [1, 0, 1], [10, 10, 0]]
A100 = [[0, 1, 1], [1, 0, 1], [100, 100, 0]]

# Define the number of samples and the dimension of the random vector X
n_samples = 10000
n_features = 3

# Define the class priors
priors = [0.3, 0.3, 0.4]

# Define the Gaussian components for the class-conditional pdfs
g1 = GaussianMixture(n_components=1, covariance_type='full', init_params='random')
g2 = GaussianMixture(n_components=1, covariance_type='full', init_params='random')
g3 = GaussianMixture(n_components=2, covariance_type='full', init_params='random')
g3.weights_ = [0.5, 0.5]

# Fit the Gaussian components to the data
g1.fit([5, 7, 3]+10*np.random.randn(10000, n_features))
g2.fit(10*np.random.randn(10000, n_features))
g3.fit([-4,-2,-3]+10*np.random.randn(10000, n_features))

# Generate the samples from the data distribution
samples = np.zeros((n_samples, n_features))
labels = np.zeros(n_samples)

for i in range(n_samples):
    label = np.random.choice([1, 2, 3], p=priors)
    if label == 1:
        samples[i] = g1.sample()[0]
    elif label == 2:
        samples[i] = g2.sample()[0]
    else:
        samples[i] = g3.sample()[0]
    labels[i] = label
    
# Implement the ERM Classification Rule using loss matrices A10 and A100
# Classify the 10K samples and count the samples corresponding to each decision-label pair
# Save the correct and incorrect points in separate vectors for each decision
confusion_matrix10 = np.zeros((3, 3))
confusion_matrix100 = np.zeros((3, 3))

# A10

correct1 = []
incorrect1 = []
correct2 = []
incorrect2 = []
correct3 = []
incorrect3 = []
for i in range(n_samples):
    sample = samples[i]
    label = labels[i]
    if g1.score(sample.reshape(1, -1)) + A10[0][int(label) - 1] < g2.score(sample.reshape(1, -1)) + A10[1][int(label) - 1] and g1.score(sample.reshape(1, -1)) + A10[0][int(label) - 1] < g3.score(sample.reshape(1, -1)) + A10[2][int(label) - 1]:
        decision = 1
    elif g2.score(sample.reshape(1, -1)) + A10[1][int(label) - 1] < g1.score(sample.reshape(1, -1)) + A10[0][int(label) - 1] and g2.score(sample.reshape(1, -1)) + A10[1][int(label) - 1] < g3.score(sample.reshape(1, -1)) + A10[2][int(label) - 1]:
        decision = 2
    else:
        decision = 3
    confusion_matrix10[int(label) - 1, int(decision) - 1] += 1
    if label == decision:
        if decision == 1:
            correct1.append(sample)
        elif decision == 2:
            correct2.append(sample)
        else:
            correct3.append(sample)
    else:
        if decision == 1:
            incorrect1.append(sample)
        elif decision == 2:
            incorrect2.append(sample)
        else:
            incorrect3.append(sample)
confusion_matrix10 = confusion_matrix10 / n_samples

# Plot the samples in 3-dimensional space
# Decision 1: circle marker
# Decision 2: triangle marker
# Decision 3: square marker
# Correct Decisions: Green // Incorrect Decisions: Red

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(np.array(correct1)[:, 0], np.array(correct1)[:, 1], np.array(correct1)[:, 2], marker='o', color='g')
ax.scatter(np.array(incorrect1)[:, 0], np.array(incorrect1)[:, 1], np.array(incorrect1)[:, 2], marker='o', color='r')
ax.scatter(np.array(correct2)[:, 0], np.array(correct2)[:, 1], np.array(correct2)[:, 2], marker='^', color='g')
ax.scatter(np.array(incorrect2)[:, 0], np.array(incorrect2)[:, 1], np.array(incorrect2)[:, 2], marker='^', color='r')
ax.scatter(np.array(correct3)[:, 0], np.array(correct3)[:, 1], np.array(correct3)[:, 2], marker='s', color='g')
### The algorithm does not make incorrect L3 guesses due to the penalty ###
#ax.scatter(np.array(incorrect3)[:, 0], np.array(incorrect3)[:, 1], np.array(incorrect3)[:, 2], marker='s', color='r')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title('A10')
plt.show()

# Plot the confusion matrix
fig = plt.figure()
ax = fig.add_subplot(111)
ax.matshow(confusion_matrix10, cmap=plt.cm.Blues)
for (i, j), z in np.ndenumerate(confusion_matrix10):
    ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
plt.title('Confusion Matrix A10')
plt.show()
    
# A100

correct1 = []
incorrect1 = []
correct2 = []
incorrect2 = []
correct3 = []
incorrect3 = []

for i in range(n_samples):
    sample = samples[i]
    label = labels[i]
    if g1.score(sample.reshape(1, -1)) + A100[0][int(label) - 1] < g2.score(sample.reshape(1, -1)) + A100[1][int(label) - 1] and g1.score(sample.reshape(1, -1)) + A100[0][int(label) - 1] < g3.score(sample.reshape(1, -1)) + A100[2][int(label) - 1]:
        decision = 1
    elif g2.score(sample.reshape(1, -1)) + A100[1][int(label) - 1] < g1.score(sample.reshape(1, -1)) + A100[0][int(label) - 1] and g2.score(sample.reshape(1, -1)) + A100[1][int(label) - 1] < g3.score(sample.reshape(1, -1)) + A100[2][int(label) - 1]:
        decision = 2
    else:
        decision = 3
    confusion_matrix100[int(label) - 1, int(decision) - 1] += 1
    if label == decision:
        if decision == 1:
            correct1.append(sample)
        elif decision == 2:
            correct2.append(sample)
        else:
            correct3.append(sample)
    else:
        if decision == 1:
            incorrect1.append(sample)
        elif decision == 2:
            incorrect2.append(sample)
        else:
            incorrect3.append(sample)
confusion_matrix100 = confusion_matrix100 / n_samples

# Plot the samples in 3-dimensional space
# Decision 1: circle marker
# Decision 2: triangle marker
# Decision 3: square marker
# Correct Decisions: Green // Incorrect Decisions: Red

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(np.array(correct1)[:, 0], np.array(correct1)[:, 1], np.array(correct1)[:, 2], marker='o', color='g')
ax.scatter(np.array(incorrect1)[:, 0], np.array(incorrect1)[:, 1], np.array(incorrect1)[:, 2], marker='o', color='r')
ax.scatter(np.array(correct2)[:, 0], np.array(correct2)[:, 1], np.array(correct2)[:, 2], marker='^', color='g')
ax.scatter(np.array(incorrect2)[:, 0], np.array(incorrect2)[:, 1], np.array(incorrect2)[:, 2], marker='^', color='r')
ax.scatter(np.array(correct3)[:, 0], np.array(correct3)[:, 1], np.array(correct3)[:, 2], marker='s', color='g')
### The algorithm does not make incorrect L3 guesses due to the penalty ###
#ax.scatter(np.array(incorrect3)[:, 0], np.array(incorrect3)[:, 1], np.array(incorrect3)[:, 2], marker='s', color='r')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title('A100')
plt.show()

# Plot the confusion matrix
fig = plt.figure()
ax = fig.add_subplot(111)
ax.matshow(confusion_matrix100, cmap=plt.cm.Blues)
for (i, j), z in np.ndenumerate(confusion_matrix100):
    ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
plt.title('Confusion Matrix A100')
plt.show()