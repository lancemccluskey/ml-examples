import * as knnClassifier from '@tensorflow-models/knn-classifier';
import { getKnnData, convertFeaturesToTensor } from './utils.js'

const data = getKnnData('./data/cars.csv');

const {
  features,
  labels,
  testFeatures,
  testLabels } = convertFeaturesToTensor(data, ['horsepower', 'weight', 'mpg'], ['passedemissions'], 25);

const classifier = knnClassifier.create();
console.log(labels.length);
console.log(features.shape[0]);
for (let row = 0; row < features.shape[0]!; row++) {
  // example: tensor, label: number|string
  const featureRow = features.slice(
    [row, 0], // 0 index of first row
    [1, -1] // 1 row, whole row
  );

  classifier.addExample(featureRow, labels[row][0]);
}

console.log(`Labels: ${classifier.getNumClasses()}`);
console.log(`Failed Emissions Examples: ${classifier.getClassExampleCount()[0]}`);
console.log(`Passed Emissions Examples: ${classifier.getClassExampleCount()[1]}`);

const accuracy = {
  correct: 0,
  incorrect: 0
};

for (let row = 0; row < testFeatures.shape[0]!; row++) {
  const testFeatureRow = testFeatures.slice(
    [row, 0],
    [1, -1]
  );

  // input: tensor, k: number
  const prediction = await classifier.predictClass(testFeatureRow);

  if (parseInt(prediction.label) === testLabels[row][0]) accuracy.correct++;
  else accuracy.incorrect++;
}

console.log('Correct Predictions: ', accuracy.correct);
console.log('Incorrect Predictions: ', accuracy.incorrect);

console.log('Accuracy: ', accuracy.correct / (accuracy.correct + accuracy.incorrect));
