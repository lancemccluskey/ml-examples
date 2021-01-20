import { LogisticRegression } from './LogisticRegression.js';
import { getLogRegData, splitData } from './utils.js';
import plot from 'node-remote-plot';

const data = getLogRegData('./data/cars.csv');

const {
  features,
  labels,
  testFeatures,
  testLabels
} = splitData(data, ['horsepower', 'weight', 'mpg'], ['passedemissions'], 25);

const logisticRegression = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  epochs: 100,
  batchSize: 50,
  decisionBoundary: 0.53
});

logisticRegression.train();

const accuracy = logisticRegression.test(testFeatures, testLabels);

console.log(accuracy);

plot({
  x: logisticRegression.costHistory.reverse()
});
