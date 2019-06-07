# AQM 2019

This repo contains short projects, useful information and various tidbits collected throughout the AQM sessions.

## ToC

[Topics](#topics)

[Data](#data)

[Notebooks](#notebooks)

[Assignment3](#assignment3)

[Tableau Examples](#tableau-examples)

[Resources](#resources)

[Images](#images)

## Topics

The main topics covered have been:

1. Exploratory Data Analysis and Visualization
    * Correlations
    * Python Plotting
2. Generalized Linear Models
    * Linear Predictor
    * Link Function (relate predictor to expectation of response)
    * Probability Distribution (relate probability to expectation)
3. Linear and Logistic Regression
    * Logits and logistic function
    * QQ Plots
    * R^2, RSE, t/F-scores, p-values, etc.
4. Classification Metrics
    * Accuracy
    * ROC, AUC
    * Confusion Matrix
5. Regularization Techniques:
    * Ridge and Lasso Regression
    * K-Folds Cross Validation
6. Databases
    * SQL Queries
    * Select, From, Where, Join, etc
7. Group Projects: Supervised Learning
    * Support Vector Machines (SVMs)
      * Works well on high feature, low example space
      * Finds maximal hyperplane separating data
      * Does not easily extend to multiple classes
      * Use of kernels to easily extend feature space
    * Random Forests and Decision Trees
      * Most human readable/interpretable method
      * Non-robust (this good for ensemble learning)
      * Works to increase the information gain at each decision
      * Random forests work to insert randomness and reduce overfitting
    * Linear/Quadratic Disciminant Analysis
      * Requires non-categorical data (based upon normally distributed features)
      * LDA approximates the Bayesian decision boundary for normal features
      * Often used for dimensionality reduction
 8. Ensemble Learning
    * Bagging = Vote Based
    * Stacking = Learning Based
 9. Unupervised Learning
    * Clustering
      * K-Means
      * Hierarchial
      * (H) Density based
    * Dimensionality Reduction
      * Linear: PCA, LDA
      * Nonlinear
    * Hungarian Algorithm
      * Perform stability analysis on clusters
10. Group Projects: Unsupervised Learning
    * Natural Language Programming (NLP)
      * Latent Dirichlet Allocation (LDA)
      * Non-Negative Matrix Factorization (NMF)
    * Time-Series Clustering
      * Dynamic Time Warping
      * Fourier Transforms
      * Wavelet Analysis
    * Quantitative Data
      * Density based clustering (DBSCAN)
      * Self-Organizing Maps (SOM)
11. Neural Networks
    * Forward/Backwards Propagation
    * Gradient Descent with Momentum
    * Cost Functions, Optimizers, Activation Functions
    * Regularization and dropout
12. NN and Keras
    * Types of optimizers, losses, activations, etc. available.
    * Sequential Model
    * Functional Model
13. NN and TensorFlow
    * How to create a simple neural net without crying.
    * Graphs, sessions, tensors, optimizers, etc.
14. Advanced NNs
    * Convolutional Neural Nets (CNN)
      * Computer Vision
    * Recurrent Neural Nets (RNN)
      * Time Series (Long Short Term Memory)
15. Group Projections: Neural Networks
    * Time Series Prediction
      * LSTM Networks
      * 1D Convolutional NNs
      * Importance of properly scaling data
    * Computer Vision
      * CNN Networks
      * Transfer Learning: USE what people have built before you
      * Tip: the 'Inception' CNN papers have useful tips for implementation
    * Natural Language Processing (Sentiment Analysis / Text Classification)
      * LSTM Networks
      * Use an embedding layer (like training word2vec) to convert text to numerical vectors
16. Tableau
    * Learning how to visualize data with many different options.
17. Light GBMs (Gradient Boosted  & Shapely Values
    * Popular boosted decision tree method.
    * Shapely values used to create visualizations of feature importance.
18. ARIMA (Time Series Forecasting)
    * Autoregressive Integrated Moving Average Model
    * Method to fit time series using moving averages, integrated (to create stationary data by differencing) and lagged observations.

## [Data](Data)

Contains all the datasets used in the problem sets. These include:

* test_eda.csv, train_eda.csv, and data_description_eda.txt
  * This is a housing data set.
  * Used in PSO_EDA
* Advertising.csv
  * Sales as a function of advertising costs in Radio, TV, and Newspaper
  * Used in PS1_LinearRegression
* social_network_data.csv
  * Classification on whether a purchase will happen based on age and salary.
  * Used in PS2_LogisticRegression
* spambase.csv and spambase_2.csv
  * Classification on whether email is spam or not.
  * Used in PS3_MetricsValidation
* BreastCancer.csv
  * Classification on whether a patient will have breast cancer
  * Used to test the DIY SVM Model
* MiniBooNE_PID.txt
  * Classification project on data from the neutrino miniboone experiment.
  * Used for the SVM and ensemble analysis.
* data_banknote_authentication.txt
  * Banknote authentication data
  * Used in PS7_NN_Keras
* Concrete_Data.csv
   * Data about... concrete.
   * Used in PS8_LinReg_TF
* titanic3.csv
   * Data pertaining to the survivors of the titanic.
   * Used in PS9_LightGBM_Shapely
* portland-oregon-average-monthly-.csv
   * Data that displays the average bus ridership in Portland over multiple years.
   * Used in PS10_ARIMA

## [Notebooks](Notebooks)

Contains useful notebooks that answer the problem sets. These include:

* PS0_EDA.ipynb
  * Contains basic exploratory data analysis
  * Covers seaborns, histograms, correlations, quantitative vs. qualitative data
* PS1_LinearRegression_Py/R.ipynb
  * Contains a Python/R implementation of ordinary least squares via gradient descent.
  * Includes basic analysis with various statistical packages (QQ plots, p-values, etc).
* PS2_Worksheet.ipynb
  * Introduces logistic regression in the GLM framework.
  * Includes probability theory for binomial and bernoulli variables.
* PS2_LogisticRegression.ipynb
  * Python implementation of logistic regression via Newton-Rhapson
  * Compares results with a scikit-learn model
* PS3_MetricsValidation.ipynb
  * Implements logistic regression using k-fold CV.
  * Logistic regression is done using ridge regression, with k-folds CV to choose best hyperparameter.
* PS4_Ensembles.ipynb
   * Implements an ensemble of learners to predict outcomes
   * Learners used include SVM, Random Forests, and LDA/QDA.
* PS5A_IrisClustering.ipynb
   * Implements various clustering models on the IRIS data to compare and introduce the methods.
   * Includes kmeans, agglomerative, gaussian mixture model, dbscan, hdbscan
* PS5B_DIY_KMeans.ipynb
   * Implements a DIY version of K-means
   * Uses K-means++ initialization and standard Euclidean distance
* PS6_DIY_NeuralNetwork.ipynb
   * Implements a DIY version of a NN with various activation functions, costs, momentum.
   * Tests the NN on various simple data sets (XOR function)
* PS7_NN_Keras.ipynb
   * Implements introductory neural networks using Keras.
   * Builds a visual aid, sequential model, and functional model. Saves them to /PS7_Keras_Models.
* PS7_TensorFlow_Guide.ipynb
   * Notebook going through the tutorial code in the low-level-API introduction.
   * Also created /TF_Example_SaveModel_* directories as a result of saving example models.
* PS8A_LinReg_TF.ipynb
   * Notebook going through the tutorial on linear regression in neural networks.
   * Implements various different versions of linear regression: different regression, batches, optmizers, etc.
* PS8B_NN_TF.ipynb
   * Example notebook implementing a simple ANN.
   * Goes through the tutorial first, before adding different metrics, custom loss function, and different initialization.
* PS9_LightGBM_Shapely.ipynb
   * Example notebook that uses the LightGBM model.
   * Also uses Shapely values to look at the most important features/factors.
* PS10_ARIMA.ipynb
   * Example notebook that explores the ARIMA timeseries method.
   * Explores timeseries stationarity, seasonality, and forecasting.
* DIY_SVM.ipynb
   * Implements a DIY version of both gaussian and linear SVMs
   * Also contains modules for k-folds, bootstrapping, and data analysis
* DIY_LDA.ipynb
   * Implements a DIY version of latent dirichlet allocation (both Gibbs sampling and Bayes inference)
   * Also includes basic Dirichlet explanations
* LSTM_Example.ipynb
   * Follows an online example to create a LSTM NN. Saves models in /LSTM_Models.
   * Also uses the data generating class instead of directly importing data.
* Hungarian_Algorithm_Example.ipynb
   * Example notebook that shows the Hungarian algorithm in practice.
   * Just uses an sklearn dataset.

## [Assignment3](Assignment3)

Contains solutions for the more involved third assigment:

* Regularization and Condition Number
* Elastic Net and Variable Selection
* Expectation Maximization (EM) & Gaussian Mixture Models (GMM)

## [Tableau Examples](Tableau_Examples)

Contains some basic Tableau examples.

## [Resources](Resources)

Contains useful reference texts, and notes from the classes.

## [Images](Images)

Contains images used in some of the notebooks.
