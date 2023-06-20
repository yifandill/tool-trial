### Goal
Implement a customized decision tree for binary classification with specified demand.

### Demand
1. In each node, use only one feature, which should be the "best" feature for splitter.
2. In the following branches and nodes, deprecate the used features above, and pick from the "remaining" features for splitter.
3. Stop until no features left for further split, or full split has been achieved beforehand.

### Implementation

#### Packages
`numpy` and `pandas` for data manipulation and storing structure.

`sklearn` for packaged machine learning tools,

namely, `sklearn.model_selection.train_test_split()`, `sklearn.tree.DecisionTreeClassifier()`.

`anytree` for packaged tree structure and tools, 

namely, `Node()` for tree construction and data storing, `RenderTree()` for tree iteration and rendering.

#### Remarks
