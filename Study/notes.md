**Machine Learning**: Teaching computers to perform tasks from past experience(data).
       Voice recognition, spam detection, fraud detection, stock market, self driving cars.

1. Navie Bayes: Based on the probability of success in the existing data, we can predict the new data.
2. Gradient Descent: What is the better way to move our prediction line to reduce the error.
3. Linear regression - best line through the points
4. Logistic regression - minimize the sum of Error function for wrongly classified data
5. Super vector Machines - place the line that increases the minimum distance of points to the line.
6. Neural Network: It is dividing the data into regions. Network of the decision chain.
7. Kernel Method: dividing with a plane or coming up with a curve
8. K-Mean clustering: Place the ClusterCenters randomly and find the distance to their closest points, move the Centers to have the less distance and follow the step to include the new points that are close to other centers and repeat it.
9. Hierarchal Clustering: Here the distance of how far we  can go to include in a cluster, determines the number of clusters.



Metrics:
    Confusion Matrix - TruePositives, TrueNegatives, FalsePositives, FalseNegatives
    Accuracy - (Points that predicted as true) / (total true points), it is not always a good measure s it gives equal weightage in confusion matrix
    Precision - ok with FalseNegatives. If I say positive it better be positive. Any FalsePositives is :O  (Finding Spam emails)
    Recall - ok with FalsePositives. If I say negative it better be negative. Any FalseNegative is :O (Diagnosing sick people)
    Precision is true column => TruePositive / (TruePositive + FalsePositive)
    Recall is false column => TruePositive / (TruePositive + FalseNegative)
    F1 score - Hormonic mean of Precision and Recall => 2 * Precision * Recall / (Precision + Recall)
    F beta Score - (1+ N^2)Precision * Recall / (N^2 Precision + Recall)
    Receiver Operator Characteristic Curve - Get (TruePositive, FalsePositive) points and the area under that curve is the metric for good split.
    Regression metrics - Mean absolute error, Mean square error(MSE), R2 Score = 1 - MSE(Linear regression model) / MSE(Simple model)

Types of errors:
    Underfitting(error due to bias) - This over simplifies the solution and it also makes mistake on training set.
    Overfitting(error due to variance) - The modal is too correct that it is more like memorizing, it does not do well with new data.
    Both performs bad on testing data. Only good model performs well on testing data.

    Model complexity Graph - Graph of errors plotted for training and testing data for different models.

    Cross Validation: For making decisions between different models, we should not use testing data, we should use Cross Validation data.

    K-fold cross validation: To best use our testing data for training, we can use K-fold method. Our data will be divided in to k-buckets
        and each time a bucket will be the testing data and get the average of the results.

    Learning Curves: Graph plotted on training data count, error value for multiple models. This helps in determining if the model is
        under-fitting or over-fitting, good model


Summary: We train the model using training data and compare the models using cross validation data and test the models using testing data.



Supervised Learning: Where the data is labeled examples, where we feed data to model about the success or failure.
Examples: Neural Networks, ensemble learning, regression, classification. Self driving cars are example of supervised classification.

Note: To determine if a problem can be solved using Supervised learning is if the data has labeled features.
One things that cannot solve using supervised learning is, clustering. We don't know how may types we are looking for.

-> Plotting the data on the graph using scattering helps in giving some information even before applying ML Algorithm.

Regression: Mapping continuous inputs to outputs (data can be discrete or continues, regression talks about continuous input)
Example: The height of the children regresses, there will be some noice where exception of being height than parents.

    Usually the models will have errors mainly because of un-modeled influences.
    We can have vector inputs for the housing problem with all features that are important.
    Linear regression and Polynomial regression are done using :  Xw = Y the value w = (XT * X)^-1  * (XT * X) Y
    Best constant in terms of squared error is mean.

    1) Parametric Regression: Y = m2 X^2 + mX + b
    2) K nearest neighbor(KNN): (data centric approach or intense based approach) - We keep the data until we need to query.
        Suppose we get 3 nearest historical data points for this query and use them to estimate.
    3) Kernel Regression: Main difference between kernel regression and KNN is in kernel, we weight the each data point based on
    its distance to the query point.

    Parametric vs non-parametric: the cannon ball distance can be best estimated using a parametric model, as it follows a well-defined trajectory(biased).
    On the other hand, the behavior of honey bees can be hard to model mathematically. Therefore, a non-parametric approach would be more suitable(unbiased).
    -> As we already know the form of equation, we aim at that bias
    -> In parametric approach we don't need to store the original data, we cannot easily update the model as more data is gathered.
    To update we need to rerun the algorithm. Then the training is slow, but querying is fast.
    -> Non Parametric needs the data to be stored in the model, hard to apply for huge datasets. But new evidence can be added easily.
    Training is fast, but querying is slow. Usually any complex model that does not follow a pattern like linear or quadratic will best solved by this.

Continuous supervised learning: (If the output is continuous its is called continuos supervised learning.)
Example: Predicting the weight of the person from the height.
Note: If there is an order of outcomes where you can scale on low to heigh, it is continuous. If we dont have any order it is discrete.

Types of errors we can have on regression:
    error = actual net worth - predicted net worth
    best regression minimizes the sum of squared errors.
    In Linear Regression in Sklearn to find m and c in y = mx +c that minimizes the sum of squares is (Ordinary Least Squares, gradient descent)
    There can be multiple lines to minimize the absolute error, where as only one line can minimize the squared errors.

Classification vs Regression:

        Property        Supervised Classification           Regression

        Output type      Discrete (class labels)            Continuous(number)

        What are you      Decision boundary                 "Best fit line"
        trying to find

        Evaluate            Accuracy                         "Sum of squares" or R^2

Decision tree for OR function is Linear, vs XOR it is full binary tree (Which has 2^n - 1) nodes, this is exponential.
Expressiveness: The number of decision trees possible with n -attributes boolean attributes with boolean output is
(possible attribute combinations - 2^n) with gain 2 possible outputs. So it is => (2)^(2^n)

ID3 algorithm: (A top down learning algorithm) uses best attribute
Best attribute is having best Gain(S,A) = Entropy(S) - Sigma |Sv| / |S| * Entropy(Sv)
Induction Bias: 1) Good splits at top of tree 2) Correct tree vs incorrect  3) Shorter trees
Note: For discrete valued attributes it does not make sense to repeat the same attribute on a specific path.
For continuos attributes it make sense, not the same question, but a different question.
Pruning: On the model tree, by collapsing the leaves, how does it creates the error on the validation set.

Modeling using Decision tree: Min-sample-split is used for fine tuning the model for better

Entropy: Measure of impurity in bunch of examples. If we have an equal distribution, we will have maximum entropy as we have maximum impurity.
Information Gain: Entropy of parent - [Weighted average] entropy of children.
Decision tree will maximize the information gain.

Note: While using Decision Tress we have to careful in picking the tuning parameters, it can easily overfit.
    We need to find a stopping point after certain height of decision tree.


Neural networks:

Initial perception of artificial neural networks (perceptron): activation (Weighted sum of inputs) > threshold
perceptron is always a linear equation, that forms half planes.
And, OR, Not are all expressible using perceptron units.

Xor can be formed by doing AND first (x1 AND x2), later do the OR of (x1 OR x2 OR x3)

The weights of perceptron are (1, -2, 1) and threshold = 1

The two rules used for finding weights are:
    1) perception rule - generates thresholded outputs
    2) gradient descent or delta rule  - generates un-thresholded outputs

Perception Rule:     Wi = wi+ delta(wi)
                    delta(wi) = learning rate(n)* (actual output y - current output y^) * Xi
    If there is a linearly separable solution, it will find if that line exist in finite number of iterations.
    algorithm stops when delta(wi) is zero. It only converges the linearly separable solutions.

Gradient descent:    delta(wi) = n * (y -a) * xi
                robust, converge local optimum
We cannot use the difference of (y, y^) in gradient descent as it is not differentiable.
To make it more smooth curve in the variation, we can use sigmoid

Neural network: Network of sigmoid units that converts input to output using some middle layers.
This mapping from input to output is differentiable means how any change of weights in the network will make the
mapping of input to output.
Back propagation: Computationally beneficial organization of the chain rule.

Disadvantages of gradient decent: It can get stuck in local minimum.
Other methods: momentum, higher order derivatives - looks at derivative of combination of weights (hamilton),
                randomized optimization, penalty for complexity
Restriction bias tells about representation power, set of hypothesis we consider.
                                perceptron: half spaces
                                Boolean: network of threshold-like units
                                Continuos: If connected and no jumps - using single hidden layer
                                Arbitrary: Stitch together - 2 layers
                    Danger of overfitting as we can have any arbitrary networks.
Preference biases: Algorithm selection of one representation over other
            Gradient descent , how to set initial weights ?
                - small set of random values - to avoids local minimum, variability of not stuck in if we run multiple times.
                    We prefer small numbers means simpler complexity over complex
            Occam's razor: entities should not be multiplied unnecessarily


Support Vector Machines:  Maximizes distance to nearest point.  MARGIN -> robustness

Optimization problem for finding max margins is a Quadratic problem.
SVM is based on only subset of data.
Mercer condition: It acts like a distance or like a similarity, our kernel function has to be a well behaved distance function,
SVM decision boundary can also be non linear
Kernel trick:  Take a non separable input say x, y and by using kernels get the x1, x2, x3, x4 as separable solutions.
XT * y => K(X,Y) Domain Knowledge

Parameter in machine learning: Kernel(linear, poly, sigmoid, rbf), C, gamma
    C - Controls tradeoff between Smooth decision boundary and classifying training points correctly
            more C value more the training data will fit, this could cause over fitting
    gamma - defines how far the influence of a single training example reaches
                        low values  - far (more smoother) - risk of under-fitting
                        high values - low  (Jaggy curve) - risk of Overfitting
SVM work well in complicated domains, where there is a clear margin of separation.
They don't perform very well on large datasets as the training time will be in order of cubic to size data set.
They don't work well if we have lots of data noice, so when classes are very overlapping you have to count independent evidence.
In this scenario Naive Bayes classifier works well


Instance based Learning: In all the above supervised learning techniques, we constructed a model function from our data
    and through away all the data, if new predictions have to be made, we are just using that function to predict.
    It is a lazy learning

    In this method, we will store all the data points and if new data comes , we will lookup the data.
        - positives: remembers, fast, simple
        - negatives: no generalization, it overfits as it memorizes,
        Among the k nearest neighbors, for getting the
                    Classification: Mode of my nearest Yi values
                    Regression: Mean of Yi
Preference bias for KNN:
                Locality - near points are similar  - thus the distance function matter a lot in defining that distance
                Smoothness - averaging
                All features matter equally

Curse of Dimensionality in ML:
    As the number of features or dimensions grows, the amount of data we need to generalize accurately grows exponentially.
        - choice of distance function matters alot - d(x, q) = Euclidean, Manhattan (weighted), Mismatches,
        - K = n, weighted average
            -> Locally weighted regression: may be a weighted regression by picking few points based on distance
Note: For picking K, distance function Domain knowledge matters.
KNN can handle both classification or regression, locally weighted Regression. Domain knowledge matters a lot.


Naive Bayes:

The Gaussian NB
It is called Naive Based, as it only looks for the frequency of the words, not considering the order of words.
Bayesian Learning: Minimum Description length.
The best hypothesis function is one that minimizes the mis classification error with less size of h.
Basyan Networks:
Note:   Factorial representation helps if we have more labels.

Conditional Independence: P(X/ y,z) = P(X/Z)

Belief Networks: (Bayes Nets, Bayesian Network, Graphical models)
    If we have conditional independence, it will be not much complex. This Belief network can grow more exponential.
    This is not a tree more like a graph.
Joint Distribution:
If we have the directed graph the topological order is used for dependency identification. The graph must be acyclic.
The distribution does not hoes exponential here, only the nodes that depends on few parents will have the dependency order in
O(2^parents)

Approximate Inference: By sampling we can understand, why cant exact ? Exact : hard, approximate : faster
Inference Rules:
    Marginalization: P(X) = Sigma (P(X,Y))
    Chain Rule: P(X,Y) = P(X) * P(Y/X)
    Bayes Rule: P(Y/X) = P(X/Y) * P(Y) / P(X)

Naive Bayes is cool:
    Inference is cheap
    Few Parameters
    Estimate parameters with labeled data
    Connects inferences with classification
    Empirically Successful

    On unseen attribute can spoil the whole.
    Induction Bias: Smooth, all possible labels exist in average

Bayes Network - Representation of joint distribution
    Examples of using networks to compute probabilities
    Sampling as a way to do approximate inference
    In general, hard to do exact inference
    Naive Bayes, (ASSUMING THE ATTRIBUTES ARE INDEPENDENT OF ONE ANOTHER) - link to classification
      - tractible
      - gold standard
      - inference in any direction (missing attributes

Ensemble learning - Boosting
    Simple rules
    combined
   => Learn over subset of data -> generates a rule
      Combine

  => Bagging: Taking random subsets and combine by mean.
  => Boosting: Done on Hardest examples, and do some kind of weighted Mean
    + Error -> stays same when it gets to some level of training
    + Confidence -> increases as more training happens, as margin of error will keep increasing.

    - Agnostic of learner

  Additional notes on boosting: https://storage.googleapis.com/supplemental_media/udacityu/367378584/Intro%20to%20Boosting.pdf



Unsupervised Learning: Learning the system based on unlabeled data
Example: Clustering, Dimensionality Reduction
    Based on patterns of common symptoms Unsupervised learning can find diseases that we did not know they exist.

 1) K-means clustering: Initialize the centers at random place
        1) Assign the points that are closer to the centers
        2) Optimize: move the centers so that the assignments will have better average distance
        Repeat this process

 Pros and Cons:  Local minimum is be a problem in the K-Means, that is why it is always better to run this algorithm multiple
 times using n_init variable in SK-Learn and get an average.
      For a fixed data set and for fixed number of initial centroids, K-Means does not give the same answer.
      The placement of initial centroids maters a lot. (K-Means is type of Hill climbing algorithms)

2) Single Linkage Clustering:
      Just make every point as a cluster and start merging based on the inter-cluster distance.
      inter-cluster distance is distance between closest(others algorithms uses average, median) points in two clusters.
      This gives the hierarchal agglomerative structure
      Pros: Terminates fast


Soft Clustering:
      If we have a point that does not belong to 2 clusters, Like if it is between 2 clusters then that point can be shared.
   Maximum Likelihood:
    <X, Z1, Z2, ...., Zk>  hidden variables
3) Expectation Maximization:  Soft clustering idea, it goes under finding the probability assignments, moving centers.
      Mostly Gaussian distribution is used.
      Expectation: probability of late variables (Expensive - involves probabilistic inference)
      Maximization: Use those variables to estimate parameters.
     Pros: Monotonically non-decreasing likelihood, does not diverge, Works with any kind of distribution
     Cons: Does not converge (practically does, very rarely happens), Can get stuck(local optima problem)- random restart.

     K-means vs EM's : K-means will stop after certain number iteration, where as EM's will have lot of probabilities to move.


Desired properties of clustering algorithms:
    Richness:  You can have any number of clusters as your solution, your algorithms should not fix the number of clusters.
    Scale-Invariance:  Change of units does not matter, relative scaling.
    Consistency:  Shrinking the intra cluster distances and increasing the inter cluster distances does not change the clustering.
Impossible theorem: No algorithm can achieve all 3 (Richness, Scale-Invariance, consistency)

Feature Scaling: Used for normalizing the feature values.
      Algorithms that would effect by feature scaling are: SVM with RBF Kernel, K-Means Clustering

Feature Selection:
       -> Only a few matters (Interpretability & insight, Knowledge Discovery)
       -> Curse of Dimensionality(If we have more features, we need exponential data)
       -> This is an np-hard problem, exponential time
    Filtering: Takes all features and gives the fewer features using searching algorithm. Fewer set is used by learning algorithm.
            Information Gain, Variance, Entropy, Useful features, Independent/Non-Redundant
           Pros: Fast as it is looking at Isolated features
           Cons: Speed-> Isolated Features, some times you might want a features if we look in combination to other.
                 Ignores the learner
        Use inductive bias of Decision tress to choose features and use other algorithm like K-means(suffers from Curse of dimensionality,
        it does not know which features are important) for actual model using selected features.
    Wrapping: Searches Features over a subset of features asks learning algorithm to do model and give the score,
            based on the score, it will update the new set of features.
            Search: Hill Climbing, Randomized optimization,
                  Forward - Select best1, keep the best1 and search for best2 that works well with best1, continue
                  Backward - select all and find sub set combination that works well.
           Pros: considers the learning bias (model bias)
           Cons: Soo Slow
    Useless VS relevance of a feature

    Xi is strongly relevant if removing it degrades BOC-Bayes Optimal Classifier
    Xi is weakly relevant if not strongly relevant, there exist a subset of features S such that adding Xi to S improve BOC
    Other wise Xi is irrelevant
    Relevance measures effect on the Bayes optimal classifier.
    Relevance - Information about data
    Usefulness - Error on Model/Learner

PCA - Principle Component Analysis (correlation by maximizing the variance  => reconstruction)
    Measurable vs Latent features:
      Square footage, Num of rooms, School Ranking, Neighborhood safety
      Size, Neighborhood
Principle component of a dataset is the direction that has maximum variance(minimizes the information loss) when we project or compress down onto them.
      More variance of data I have along PC, higher the PC ranked.
  Example: Facial Recognition


 Feature Transformation:(Linear Transformation)
      Feature selection is subset of feature Transformation
Example: Information Retrieval (ad hoc)  Give me related keyword documents
      -> lots of words, good indicators, polysemy(false positives), synonymy(false negatives)

Independent Component Analysis: Independence   (x1, x2, ....) => (y1, y2, ...) with I(yi, yj) =0 I(Y, x) is more means no much information loss
     Blind source separation problem or cocktail party problem : 3 people talking(hidden layer) and recorded by 3 microphones.
      directional

Faces are inputed:
      PCA -> Brightness, Average Face
      ICA -> Nose, Eyes, Mouth etc

RCA- RandomComponent Analysis:
LDA - Linear Discriminant Analysis: Finds the projection that discriminates based on labels





Reinforcement Learning(decision making):
In this we attach reward and punishments to different outcomes, by weighting these reward and
punishments, we can teach the right level of priority to assign different goals. Augmenting these reward and punishments will help.

  -> mark of decision processes
  -> game theory and multi agent interactions
In supervised learning we have function approximation, unsupervised has clustering or description,
reinforcement learning  y = f(x) given x, z

Markav's Decision process: (MDP)  Bellmen equation
        State,
        action,
        model(describes the rules, probability of going to other state from current state by taking the action)
                model is the physics of the world
        reward: positive points if we reach goals, or negative points if we do mistakes,

        Rewards tell moment to moments vs (value functions)utilities tells the  long term rewards.

Solution:   Policy -> pi(S) -> a
            What action to take for any state we came across

Markovian property: only present matters, things are stationary(rules don't change)
   picking the rewards matter a lot. Rewards are our domain knowledge.
    The path we choose depends on the reward scale and the time you got before you end.

Stationarity of preferences:  Look math in video
       Discounted -> geometric
       infinite -> finite
       Delayed rewards are long term utility
Start with arbitrary utilities(value functions), update utilities based on neighbors

Examples of planner are -
    Value iteration: Get utility iteration  :  https://www.youtube.com/watch?v=doxTNCH7oHc
    Policy iteration

Note: 3 Approaches to do RL :  https://www.youtube.com/watch?time_continue=46&v=bFPoHrAoPoQ
     Policy Search, Value-function based,
Q-learning: is family of algorithms, what to use for Q cap selection,

Game Theory:

2-player, zero-sum deterministic game of perfect information - minMax solution of matrix
  for non-deterministic - still minmax works well

  but for deterministic game of hidden information - minmax = maxmin does not hold well

Prisoner dilemma:

Stochastic Games:   MDP:RL :: StochasticGames:MultiAgentRL

Zero-sum Stochastic Games:
     we got minmax-Q function
     Value iteration works,
     minMax-Q Converges,
     Unique solution,
     Policies for the 2 players can be computed independently
     Update efficiently the equation

General-sum Stochastic Games:   almost NP problem
    Nash Equilibrium instead of minMax
    Value iteration does not works,
    Nash-Q does not Converge,
    No Unique solution to Q*,
    Policies cannot be computed independently
    Update is not efficient

Other ideas:
    Repeated stochastic games
    Cheap talk -> correlated equilibrium
    Cognitive Hierarchy -> best responses
    Side payments (coco values) cooperative competitive values






  Deep Neural Networks:

    Neural network general formula: w1 * x1 + w2 * x2 + b = 0   or (W * X + b = 0) where W = (w1, w2,..) and X = (x1, x2, ..)

   Perceptron: https://www.youtube.com/watch?v=hImSxZyRiOw
        takes the inputs x1, x2, ... xn multiplied by weights and it can take 1 as input and  multiply it with bias
           w1 x1 + w2 x2 + .... + 1 * b  and our we can have node that sums the values and other node that is a step
           function checks if input is positive and returns 1. We can have bias either coming as input or value inside the node.

        Note: These perceptrons can be seen as a combination of nodes. Just takes the inputs and returns either 0 or 1.
  Perceptron trick: If we have 2 types of points and if we draw a random line, for the wrongly separated points we need to move the line closer to those points.
  https://www.youtube.com/watch?v=jfKShxGAbok

  To use the gradient descent our error function need to be continuous, i.e error function need to be differentiable.

  Error function: Instead of using the step function, if we can use sigmoid function (1/ 1+ e^-x), we can use the gradient decent
  Softmax method: To get the probability of occurrence of a multi-labeled class is done using the exponential average of its scores.
  One-hot encoding: For any input of no-numbers, we need to turn then into numbers by having one variable for each of classes.
  Maximum likelihood model: If we have 2 models, the best model is the one that gives more probabilities to the event that happen to us.
  Cross-entropy: Instead of multiplying the probabilities of the points, if we take a negative log of all vales, it represents the cross-entropy.
       For a better model, the -log(0.9) -log(0.99) = very less value compared to the bad model with high cross-entropy value.
       Maximizing the entropy to minimizing the cross-entropy
  Logistic regression: the error function for the binary classification is
      Error function E(W,b) = -1/ m Sum_i_1_to_m ((1-yi)(ln(1-y_predi)) - yi * ln(y_predi)    where y_predi = sigma (W* xi + b)
  Gradient Descent: https://classroom.udacity.com/nanodegrees/nd009/parts/99115afc-e849-48cf-a580-cb22eea2ba1b/modules/777db663-2b0d-4040-9ae4-bf8c6ab8f157/lessons/21d39927-8e4f-44ba-8425-d15889ac19e2/concepts/ca6eff40-a3e2-4d53-85f4-d2454b538d87

  Similarities between perceptron algorithm and Gradient descent algorithm:
  In gradient descent, the misclassified points tells to come closer vs correctly classified points tells to go away.

  Don't forget the sigmoid function in the neural network after adding the weighted sum of inputs and bias values.

Early stopping: It is used to stop at the best epoch level for a given problem by stoping at the first increasing point
in the model complexity graph.

The gradient at each node is the product of derivatives at each node in the path to output. So we will be getting very tiny gradient decent values.
Th fix is to get more bigger momentum changes: by using tanh function and relu function(used for continuous values)

Gradient descent does not help if we have local minimum in the way to globally optimum solution.
Solution: Random restarts, momentum(move with speed of last few steps we made)
