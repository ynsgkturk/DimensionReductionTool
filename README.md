# Dimension Reduction

Dimension reduction in machine learning is crucial for enhanced model efficiency and performance. It accelerates training, mitigates the curse of dimensionality, improves model interpretability, and reduces overfitting. By simplifying complex datasets, dimension reduction facilitates faster and more accurate predictions, making it an essential step in building robust machine learning models.

In this repository I worked on a meta-heuristic dimension reduction algorithm using Particle Swarm Optimization (PSO) and FDB-Archimedes Optimization Algorithm (FDB-AOA).

# Particle Swarm Optimization (PSO)

Particle Swarm Optimization (PSO) is a population-based optimization algorithm inspired by the social behavior of birds and fish. In PSO, individuals, or particles, traverse the solution space, adjusting their positions based on personal experience and the collective information from their neighbors. This collaborative approach enables efficient exploration and exploitation of the search space, making PSO particularly effective for solving optimization problems. With its simplicity and adaptability, PSO is widely used in various fields for finding optimal solutions and tackling complex optimization challenges.

# Archimedes Optimization Algorithm 

The AOA algorithm is a population-based algorithm. In the AOA algorithm, population individuals are objects floating in liquid. Similar to other population-based meta-heuristic algorithms, AOA generates a random initial population with random volume, density, and acceleration, initiating the search process life cycle. At this stage, each object is initialized at a random position in the liquid. After evaluating the fitness of the initial population, the AOA algorithm continues to run until the termination condition is met. In each iteration of the AOA algorithm, the density and volume of objects are updated. The acceleration of the object is updated based on collision situations with any neighboring object. The updated density, volume, and acceleration are used to determine the new position of that object [1].
FDB-AOA [2] algorithm is improved version of AOA utilizing a powerful selection method Fitness Distance Balance (FDB) [3].

# Experimental Study

In this section, I used machine learning algorithms to see the performance of dimension reduction on Urban Land Cover Dataset.
Mentioned dataset contains 147 features and a target feature. 

Table 1. Parameter Settings of Algorithms.

| Algorithm | Parameter                                      | 
|-----------|------------------------------------------------|
| KNN       | k = 5                                          | 
| PSO       | Number of Particles = 25; Max Iteration = 1200 | 
| FDB-AOA   | Number of Materials = 25; Max Iteration = 1200 |


According to the settings in Table 1, K-Nearest Neighbors (KNN),
Random Forest (RF), XGBoost (XGB), Light-Gradient Boosting (LGBM), 
Hist-Gradient Boosting (HGB) algorithms executed and
got the results below in Table 2 and Table 3.

Table 2. Accuracies of Machine Learning Models

| Model | Score | PSO Score | FDB-AOA Score |
|:-----:|:-----:|:---------:|:-------------:|
|  XGB  |  85   |    83     |      76       |
|  KNN  |  45   |    43     |      82       |
| LGBM  |  87   |    87     |      80       |
|  RF   |  82   |    83     |      80       |
|  HGB  |  85   |    87     |      81       |

According to the Table 2;
* There's significant improvement on KNN algorithm.
* Random Forest is almost same with %90 fewer features.
* With 90% (133) fewer features, accuracies are close to each other.

Tablo 4. Extracted Feature Rates

|              Method              |   Train    |   Test    | Extracted Features | Extracted Feature Rate |
|:--------------------------------:|:----------:|:---------:|:------------------:|:----------------------:|
|       Original Data Shape        | (507,147)  |(168, 147) |         0          |           0%           |
|   Dimension Reduction (w/ PSO)   | (507, 130) | (168,130) |         17         |          12%           |
| Dimension Reduction (w/ FDB-AOA) | (507, 14)  | (168,14)  |        133         |          90%           |

Despite the algorithm extracts 90% of the features, machine learning models exhibit minimal loss, achieving an impressive 80% accuracy. This underscores the effectiveness of dimension reduction in streamlining the dataset without compromising the overall predictive power of the model.
# References

1. Hashim, F. A., Hussain, K., Houssein, E. H., Mabrouk, M. S., & Al-Atabany, W. (2020).
Archimedes optimization algorithm: a new metaheuristic algorithm for solving optimization
problems. Applied Intelligence. doi:10.1007/s10489-020-01893-z

2. Yenipinar, B., Şahin, A., Sönmez, Y., Yilmaz, C., & Kahraman, H. T. (2023). Design Optimization
of Induction Motor with FDB-Based Archimedes Optimization Algorithm for High Power Fan and
Pump Applications. In The International Conference on Artificial Intelligence and Applied
Mathematics in Engineering (pp. 409-428). Springer, Cham.

3. Kahraman, H. T., Aras, S., & Gedikli, E. (2020). Fitness-distance balance (FDB): A new selection
method for meta-heuristic search algorithms. Knowledge-Based Systems, 190, 105169.
