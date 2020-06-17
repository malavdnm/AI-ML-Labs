import util
import numpy as np
import sys
import random

PRINT = True

###### DON'T CHANGE THE SEEDS ##########
random.seed(42)
np.random.seed(42)

def small_classify(y):
    classifier, data = y
    return classifier.classify(data)

class AdaBoostClassifier:
    """
    AdaBoost classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    
    """

    def __init__( self, legalLabels, max_iterations, weak_classifier, boosting_iterations):
        self.legalLabels = legalLabels
        self.boosting_iterations = boosting_iterations
        self.classifiers = [weak_classifier(legalLabels, max_iterations) for _ in range(self.boosting_iterations)]
        self.alphas = [0]*self.boosting_iterations

    def train( self, trainingData, trainingLabels):
        """
        The training loop trains weak learners with weights sequentially. 
        The self.classifiers are updated in each iteration and also the self.alphas 
        """
        
        self.features = trainingData[0].keys()
        "*** YOUR CODE HERE ***"
        n = len(trainingData)
        w = np.ones(n) / n
        for i in range(self.boosting_iterations):
            self.classifiers[i].train(trainingData, trainingLabels, w)
            error = 0.0
            pred = np.array(self.classifiers[i].classify(np.array(trainingData)))
            error += w[np.array(trainingLabels) != pred].sum()
            f = error / (1 - error)
            w[trainingLabels == pred] *= f
            w /= w.sum()
            self.alphas[i] = -np.log(f)

        # util.raiseNotDefined()

    def classify( self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label. This is done by taking a polling over the weak classifiers already trained.
        See the assignment description for details.

        Recall that a datum is a util.counter.

        The function should return a list of labels where each label should be one of legaLabels.
        """

        "*** YOUR CODE HERE ***"
        ans = []
        for datum in data:
            guess=np.zeros(self.boosting_iterations)
            for j in range(self.boosting_iterations):
                guess[j] = self.alphas[j] * self.classifiers[j].classify([datum])[0]
            guess[guess==0]=np.random.choice([-1,1])
            ans.append(np.sign(np.sum(guess)))
        return ans
        # util.raiseNotDefined()