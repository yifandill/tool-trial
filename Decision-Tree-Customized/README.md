### Goal
Implement a customized decision tree for binary classification with specified demand.

### Demand
#### Training
1. In each node, use only one feature, which should be the "best" feature for splitter.
2. In the following branches and nodes, deprecate the used features above, and pick from the "remaining" features for splitter.
3. Record the exact label for the node if it has achieved full split; otherwise, continue.
4. Stop until no features left for further split, or full split has been achieved beforehand.
5. For leaf nodes without full split (in maximum depth), record the label in the majority and the frequency in probability of both labels.
6. For each leaf node, record the index of training samples, seen as a cluster.

#### Testing (Validation/Prediction)
Given a testing set of features (with/without labels), referring to the trained decision tree, return:
1. The predicted labels or according probability for each sample;
2. The cluster of each sample, in relation with the training set index.

### Packages
#### `numpy` and `pandas`
For data manipulation and storing structure.

#### `sklearn`
For packaged machine learning tools,

namely, `sklearn.model_selection.train_test_split()`, `sklearn.tree.DecisionTreeClassifier()`.

#### `anytree`
For packaged tree structure and tools, 

namely, `Node()` for tree construction and data storing, `RenderTree()` for tree iteration and rendering.

### Implementation

#### Feature Selection
Pick the best feature to use in one layer of training, according to impurity loss.

#### One-layer Training
Execute the training of one layer, by means of `DecisionTreeClassifier(splitter='best', max_depth=1, max_features=1)`.

#### Features Storage
Stored all attributes in the nodes of the tree structure, for purpose of indepedence among paths.

#### Tree Construction
Implemented via self-recursion.

Execute the training process of the branch of one parent node to its two children nodes.

