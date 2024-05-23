% 加载数据集
data = readtable('iris.csv');
data.Properties.VariableNames = {'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'};
X = data(:, 1:4);
Y = data.species;

% 随机删除10%数据
rng('default');
missingRate = 0.1;
for i = 1:height(X)
    for j = 1:width(X)
        if rand() < missingRate
           X{i, j} = NaN;
       end
   end
end

% 将数据分为训练集和测试集
cv = cvpartition(height(data), 'HoldOut', 0.2);
XTrain = X(training(cv), :);
YTrain = Y(training(cv), :);
XTest = X(test(cv), :);
YTest = Y(test(cv), :);

% 设置决策树参数，以强制使用所有特征
treeParams = {'MinParentSize', 1, 'MinLeafSize', 1, 'Prune', 'off'};

% 通过信息增益生成决策树
infoGainTree = fitctree(XTrain, YTrain, 'SplitCriterion', 'deviance', treeParams{:});

% 通过基尼指数生成决策树
giniTree = fitctree(XTrain, YTrain, 'SplitCriterion', 'gdi', treeParams{:});

% 对决策树进行剪枝
infoGainTreePruned = prune(infoGainTree, 'Level', 1);
giniTreePruned = prune(giniTree, 'Level', 1);

% 在数据缺失的情况下生成决策树
missingDataTreeInfoGain = fitctree(XTrain, YTrain, 'SplitCriterion', 'deviance', 'Surrogate', 'on', treeParams{:});
missingDataTreeGini = fitctree(XTrain, YTrain, 'SplitCriterion', 'gdi', 'Surrogate', 'on', treeParams{:});

% 在数据缺失的情况下剪枝决策树
missingDataTreeInfoGainPruned = prune(missingDataTreeInfoGain, 'Level', 1);
missingDataTreeGiniPruned = prune(missingDataTreeGini, 'Level', 1);

% 计算精度
accuracyInfoGain = sum(strcmp(predict(infoGainTreePruned, XTest), YTest)) / height(YTest);
accuracyGini = sum(strcmp(predict(giniTreePruned, XTest), YTest)) / height(YTest);

% 显示精度
fprintf('InfoGain Pruned Decision Tree Accuracy: %.2f%%\n', accuracyInfoGain * 100);
fprintf('Gini Pruned Decision Tree Accuracy: %.2f%%\n', accuracyGini * 100);

% 可视化
view(infoGainTreePruned, 'Mode', 'graph');
view(giniTreePruned, 'Mode', 'graph');
