import { getData, convertToTensor, buildBestModel, predict } from './utils.js';
import * as tf from '@tensorflow/tfjs-node';
import path from 'path';
import fs from 'fs';

const data = getData('./data/cars.csv');

const {
  features,
  labels,
  testFeatures,
  testLabels } = convertToTensor(data, ['horsepower', 'displacement', 'weight'], ['mpg'], 50);

let model: tf.Sequential;

const { mean: featuresMean, variance: featuresVariance } = tf.moments(features);

if (fs.existsSync(`${path.resolve()}/models/linear-regression/model.json`)) {
  model = await tf.loadLayersModel(`file://${path.resolve()}/models/linear-regression/model.json`) as tf.Sequential;
} else {
  model = await buildBestModel(features, labels, testFeatures, testLabels, 25, 25);
  await model.save(`file://${path.resolve()}/models/linear-regression`);
}

// 2021 Lexus RX 450 HL, MPG: 28.5
const lexusPred = predict(model, tf.tensor([[308, 211, 2.18]]), featuresMean, featuresVariance) as tf.Tensor;
lexusPred.print();

// 2021 Toyota Camry LE 4-Cylinder, MPG: 32
const camryPred = predict(model, tf.tensor([[208, 152, 1.48]]), featuresMean, featuresVariance) as tf.Tensor;
camryPred.print();

// 2005 Toyota Camry LE 4-Cylinder, MPG: 29
const oldCamryPred = predict(model, tf.tensor([[160, 145, 1.38]]), featuresMean, featuresVariance) as tf.Tensor;
oldCamryPred.print();

// 1995 Ford Bronco XLT 4WD, MPG: 14
const fordBroncoPred = predict(model, tf.tensor([[205, 302, 2.06]]), featuresMean, featuresVariance) as tf.Tensor;
fordBroncoPred.print();
