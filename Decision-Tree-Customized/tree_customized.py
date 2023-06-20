import copy
import typing
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from anytree import Node, RenderTree


class DecisionTreeCustomized:
    def __init__(self):
        """
        self.X: The set of features for training.
        self.y: The set of labels for training.
        self.X_p: The set of features for testing (validation) or prediction.
        self.y_v: The set of labels for validation. If for prediction, self.y_v = None.
        self.y_p: Records the predicted labels.
        self.y_prob: Records the probability of predicted labels.

        self.col_list: The name list of features.
        self.label: The name list of 2 labels.
        self.root: The root of the tree structure for storing data.
        self.cluster: Records the clustering result for the prediction set, in relation with the training set index.
        """
        self.X = np.array([])
        self.y = np.array([])
        self.X_p = np.array([])
        self.y_v = np.array([])
        self.y_p = np.array([])
        self.y_prob = pd.DataFrame([])

        self.col_list = list()
        self.label = tuple()
        self.root = None
        self.cluster = pd.DataFrame([])

    def set_data(self, df: pd.DataFrame, y: np.ndarray) -> None:
        """
        Prepare the data for training.
        """
        self.X = df.values.astype(np.float32)
        self.col_list = df.columns.to_list()
        self.label = tuple(np.unique(y))
        self.y = np.where(y == self.label[0], 0, 1)

    def set_predict(self, X: np.ndarray, y: np.ndarray = None) -> None:
        """
        Prepare the data for prediction or validation.
        """
        self.X_p = X
        self.y_v = np.where(y == 0, self.label[0], self.label[1])

    @staticmethod
    def node_name(name: str) -> list:
        """
        Tool function. Generate the name list for children nodes.
        Used in self.child_branch().
        """
        return [name+'l', name+'r']

    @staticmethod
    def set_node(
            name: str = None,
            parent: Node = None,
            feature_id: int = None,
            threshold: float = None,
            X: np.ndarray = None,
            y: np.ndarray = None,
            leaf_judge: np.ndarray = None,
            col_list: list = None,
            rst_tmp: np.ndarray = None,
            rst: typing.Union[int, float, str, None] = None,
            prob: list = None,
            index_tmp: np.ndarray = None
            ) -> Node:
        """
        Tool function. Generate the node of the tree structure, together with the storing attributes.
        Used in self.child_branch(), self.tree_construction().
        """
        return Node(
            name,
            parent=parent,
            feature_id=feature_id,
            threshold=threshold,
            X=X,
            y=y,
            leaf_judge=leaf_judge,
            col_list=col_list,
            rst_tmp=rst_tmp,
            rst=rst,
            prob=prob,
            index_tmp=index_tmp
            )

    def show_tree(self, prob: bool = False) -> None:
        """
        Display the trained tree, with the desired attributes.
        """
        for pre, _, node in RenderTree(self.root):
            if not node.is_leaf:
                print('%s%s: \'%s\' <= %.2f' % (pre, node.name, node.col_list[node.feature_id], node.threshold))
            else:
                num0 = node.rst_tmp[0]
                num1 = node.rst_tmp[1]
                if num0 >= num1:
                    node.rst = self.label[0]
                else:
                    node.rst = self.label[1]
                if (num0 == 0) or (num1 == 0):
                    print('%s%s: full split, %s' % (pre, node.name, str(node.rst)))
                else:
                    if prob:
                        node.prob = [num0 / (num0 + num1), num1 / (num0 + num1)]
                        print('%s%s: %.2f%% %s and %.2f%% %s' % (pre, node.name, node.prob[0] * 100, str(self.label[0]),
                                                                 node.prob[1] * 100, str(self.label[1])))
                    else:
                        print('%s%s: majority, %s' % (pre, node.name, str(node.rst)))

    @staticmethod
    def find_best_trainer(X: np.ndarray, y: np.ndarray) -> int:
        """
        Tool function. Return the index of the best feature.
        Pick the best feature to use in one layer of training, according to impurity loss.
        Used in self.training_one_layer().
        """
        score_list = []
        for i in range(X.shape[1]):
            X_tmp = copy.deepcopy(X)[:, i].reshape(-1, 1)
            clf = tree.DecisionTreeClassifier(splitter='best', max_depth=1, max_features=1)
            clf = clf.fit(X_tmp, y)
            score_tmp = clf.tree_.impurity
            score_list.append(score_tmp[0] * 2 - score_tmp[1] - score_tmp[2])
        return np.array(score_list).argmax().astype(int)

    def training_one_layer(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Tool function. Execute the training of one layer, by means of DecisionTreeClassifier.
        Return the training attributes of one layer trainer.
        Used in self.child_branch(), self.tree_construction().
        """
        feature_ind = self.find_best_trainer(X, y)
        X_tmp = copy.deepcopy(X)[:, feature_ind].reshape(-1, 1)
        clf = tree.DecisionTreeClassifier(splitter='best', max_depth=1, max_features=1)
        clf = clf.fit(X_tmp, y)

        left_ind = np.where(clf.tree_.apply(X_tmp.astype(np.float32)) == clf.tree_.children_left[0])[0]
        right_ind = np.where(clf.tree_.apply(X_tmp.astype(np.float32)) == clf.tree_.children_right[0])[0]

        # feature_ind = clf.tree_.feature[0]
        threshold = clf.tree_.threshold[0]
        leaf_judge = clf.tree_.value

        return left_ind, right_ind, feature_ind, threshold, leaf_judge

    def child_branch(self, node: Node) -> None:
        """
        Tool function. Implemented via self-recursion.
        Execute the training process of the branch of one parent node to its two children nodes.
        Stored all attributes in the nodes of the tree structure.
        Used in self.tree_construction().
        """
        left_ind, right_ind, feature_ind, threshold, leaf_judge = self.training_one_layer(node.X, node.y)
        node.feature_id = feature_ind
        node.threshold = threshold
        node.leaf_judge = copy.deepcopy(leaf_judge)

        X_tmp = np.delete(copy.deepcopy(node.X), feature_ind, axis=1)
        col_tmp = copy.deepcopy(node.col_list)
        col_tmp.pop(feature_ind)
        list_child = self.node_name(node.name)
        locals()[list_child[0]] = self.set_node(
            list_child[0],
            parent=node,
            X=copy.deepcopy(X_tmp)[left_ind],
            y=copy.deepcopy(node.y)[left_ind],
            col_list=col_tmp,
            rst_tmp=copy.deepcopy(node.leaf_judge)[1][0],
            index_tmp=copy.deepcopy(node.index_tmp)[left_ind]
        )
        locals()[list_child[1]] = self.set_node(
            list_child[1],
            parent=node,
            X=copy.deepcopy(X_tmp)[right_ind],
            y=copy.deepcopy(node.y)[right_ind],
            col_list=col_tmp,
            rst_tmp=copy.deepcopy(node.leaf_judge)[2][0],
            index_tmp=copy.deepcopy(node.index_tmp)[right_ind]
        )

        if (not (node.leaf_judge[1][0] == 0).any()) and (node.children[0].X.shape[1] > 0):
            self.child_branch(node.children[0])
        if (not (node.leaf_judge[2][0] == 0).any()) and (node.children[1].X.shape[1] > 0):
            self.child_branch(node.children[1])

    def tree_construction(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Training of the customized decision tree classifier.
        Trained result stored in every node of the tree structure.

        Note:
            The tree structure is represented by the root node of the tree, i.e., self.root.
        """
        # Parent Node
        left_ind, right_ind, feature_ind, threshold, leaf_judge = self.training_one_layer(X, y)
        self.root = self.set_node(
            'Origin',
            parent=None,
            feature_id=feature_ind,
            threshold=threshold,
            X=copy.deepcopy(X),
            y=copy.deepcopy(y),
            leaf_judge=copy.deepcopy(leaf_judge),
            col_list=self.col_list
        )

        X_tmp = np.delete(copy.deepcopy(X), feature_ind, axis=1)
        col_tmp = copy.deepcopy(self.col_list)
        col_tmp.pop(feature_ind)
        l = self.set_node(
            'l',
            parent=self.root,
            X=copy.deepcopy(X_tmp)[left_ind],
            y=copy.deepcopy(y)[left_ind],
            col_list=col_tmp,
            index_tmp=copy.deepcopy(left_ind),
            rst_tmp=copy.deepcopy(self.root.leaf_judge)[1][0]
        )
        r = self.set_node(
            'r',
            parent=self.root,
            X=copy.deepcopy(X_tmp)[right_ind],
            y=copy.deepcopy(y)[right_ind],
            col_list=col_tmp,
            index_tmp=copy.deepcopy(right_ind),
            rst_tmp=copy.deepcopy(self.root.leaf_judge)[2][0]
        )

        # Child Node: Left
        if (not (l.rst_tmp == 0).any()) and (l.X.shape[1] > 0):
            self.child_branch(l)

        # Child Node: Right
        if (not (r.rst_tmp == 0).any()) and (r.X.shape[1] > 0):
            self.child_branch(r)

    def predict_one_sample(self, x: np.ndarray) -> typing.Union[int, float, str, None]:
        """
        Tool function. Execute the prediction of one sample.
        Return the predicted label of this sample.
        Used in self.tree_predict().
        """
        node = self.root
        while node.rst is None:
            if x[node.feature_id] >= node.threshold:
                x = np.delete(x, node.feature_id)
                node = node.children[0]
            else:
                x = np.delete(x, node.feature_id)
                node = node.children[1]
        return node.rst

    def tree_predict(self) -> None:
        """
        Prediction of the testing set.
        Used when prob = False.
        """
        self.y_p = np.apply_along_axis(self.predict_one_sample, axis=1, arr=self.X_p)

    def tree_predict_prob(self) -> None:
        """
        Prediction of the testing set.
        Used when prob = True.
        """
        rst_tmp = list()
        for i in range(self.X_p.shape[0]):
            x_tmp = self.X_p[i]
            node = self.root
            while node.rst is None:
                if x_tmp[node.feature_id] >= node.threshold:
                    x_tmp = np.delete(x_tmp, node.feature_id)
                    node = node.children[0]
                else:
                    x_tmp = np.delete(x_tmp, node.feature_id)
                    node = node.children[1]
            if node.prob is not None:
                rst_tmp.append(node.prob)
            else:
                rst_tmp.append(node.rst)
        self.y_prob = pd.DataFrame(rst_tmp, columns=['Predict'])

    def tree_cluster(self) -> None:
        """
        Clustering based on the trained tree.
        """
        cluster_tmp = list()
        for i in range(self.X_p.shape[0]):
            x_tmp = self.X_p[i]
            node = self.root
            while node.rst is None:
                if x_tmp[node.feature_id] >= node.threshold:
                    x_tmp = np.delete(x_tmp, node.feature_id)
                    node = node.children[0]
                else:
                    x_tmp = np.delete(x_tmp, node.feature_id)
                    node = node.children[1]
            # cluster_tmp.append([node.index_tmp, [i]])
            cluster_tmp.append([tuple(node.index_tmp)])
        cluster_df = pd.DataFrame(cluster_tmp, columns=['Train Cluster'])
        self.cluster = cluster_df
        # cluster_df = pd.DataFrame(cluster_tmp, columns=['Train', 'Predict'])
        # cluster_df['Train'] = cluster_df['Train'].apply(tuple)
        # self.cluster = cluster_df.groupby('Train')['Predict'].\
        #     apply(lambda x: [element for l in x for element in l]).reset_index()
        # self.cluster['Predict'] = self.cluster['Predict'].apply(tuple)

    def accuracy(self) -> None:
        """
        Calculate the accuracy of the testing data, for validation.
        """
        accuracy = np.sum(self.y_p == self.y_v)/len(self.y_p)
        print('Accuracy: ', accuracy)

    def main(self,
             df: pd.DataFrame,
             y: np.ndarray,
             test_size: float = 0.2,
             random_state: typing.Union[int, np.random.RandomState, None] = None,
             prob: bool = False
             ) -> pd.DataFrame:
        """
        For training and validation, with df and y for training and testing data split.

        Parameters:
            df: The dataset of features.
            y: The dataset of labels.
            test_size: Controls the size of training and testing data split.
            random_state: Controls the randomness of the data split.
            prob: The parameter for display mode;
                if prob=False, print all labels in specific values;
                if prob=True, print probabilities for undetermined node.

        Prints:
            The visualization of trained decision tree with all branches and nodes specified.
            The predicted labels or probability for the prediction set.
            The accuracy of the testing data, for validation.
            The clustering result for the prediction set, in relation with the training set index.

        Returns:
            pd.DataFrame: The clustering result.
        """
        # Data Preparation
        self.set_data(df=df, y=y)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            test_size=test_size, random_state=random_state)
        self.set_predict(X_test, y_test)
        # Decision Tree Training
        self.tree_construction(X_train, y_train)
        # Visualization of Trained Tree
        self.show_tree(prob=prob)
        print('\n=====================================================================================================')
        # Prediction and Validation
        if prob:
            self.tree_predict_prob()
            print('Predicted labels or probability of [%s, %s]:\n' % (str(self.label[0]), str(self.label[1])),
                  self.y_prob)
        else:
            self.tree_predict()
            print('Predicted labels:\n', self.y_p)
            if self.y_v is not None:
                self.accuracy()
        print('=====================================================================================================')
        # Clustering
        self.tree_cluster()
        print('Cluster result:\n', self.cluster)

        return self.cluster

    def main2(self,
              X: pd.DataFrame,
              y: np.ndarray,
              X_new: pd.DataFrame,
              prob: bool = False
              ) -> pd.DataFrame:
        """
        For prediction, with no y labels for validation.

        Parameters:
            X: The training set of features.
            y: The training set of labels.
            X_new: The set of features for prediction.
            prob: The parameter for display mode;
                if prob=False, print all labels in specific values;
                if prob=True, print probabilities for undetermined node.

        Prints:
            The visualization of trained decision tree with all branches and nodes specified.
            The predicted labels or probability for the prediction set.
            The clustering result for the prediction set, in relation with the training set index.

        Returns:
            pd.DataFrame: The clustering result.
        """
        # Data Preparation
        self.set_data(df=X, y=y)
        self.set_predict(X_new.values.astype(np.float32))
        # Decision Tree Training
        self.tree_construction(self.X, self.y)
        # Visualization of Trained Tree
        self.show_tree(prob=prob)
        print('\n=====================================================================================================')
        # Prediction
        if prob:
            self.tree_predict_prob()
            print('Predicted labels or probability of [0, 1]:\n', self.y_prob)
        else:
            self.tree_predict()
            print('Predicted labels:\n', self.y_p)
        print('=====================================================================================================')
        # Clustering
        self.tree_cluster()
        print('Cluster result:\n', self.cluster)

        return self.cluster


if __name__ == '__main__':
    '''test: breast_cancer dataset'''
    # from sklearn.datasets import load_breast_cancer
    # bc = load_breast_cancer()
    # X_df = pd.DataFrame(bc.data[:, :4], columns=bc.feature_names[:4])
    # y_array = bc.target
    # TD = DecisionTreeCustomized()
    # rng = np.random.RandomState(42)
    # TD.main(X_df, y_array, test_size=0.3, random_state=rng, prob=False)

    '''test: train.xlsx'''
    data = pd.read_excel('./train.xlsx', index_col=0)
    X_df = data[data.columns[-2:]]
    y_array = data[data.columns[0]].values
    TD = DecisionTreeCustomized()
    TD.main(X_df, y_array, test_size=0.1, random_state=8, prob=True)
