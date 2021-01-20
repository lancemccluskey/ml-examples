import fs from 'fs';
import * as tf from '@tensorflow/tfjs-node';
import Papa from 'papaparse';
import shuffleSeed from 'shuffle-seed';

export function getData(file: string) {
  const { data, errors, meta } = Papa.parse(
    fs.readFileSync(file, { encoding: 'utf8' }), {
      header: true,
      dynamicTyping: true
  });

  if (errors.length > 0) {
    throw new Error(`Error parsing file: ${file}`);
  }

  return data;
}

export function convertToTensor(data: any[], featureColumns: string[], labelColumns: string[], testSplit: number) {
  return tf.tidy(() => {
    // tf.util.shuffle(data);
    data = shuffleSeed.shuffle(data, 'phrase');

    const features = data.map(d => featureColumns.map(feature => d[feature]));
    const labels = data.map(d => labelColumns.map(label => d[label]));

    const featureTensor = tf.tensor(features);
    const labelTensor = tf.tensor(labels);

    return {
      features: featureTensor.slice(testSplit),
      labels: labelTensor.slice(testSplit),
      testFeatures: featureTensor.slice(0, testSplit),
      testLabels: labelTensor.slice(0, testSplit),
    }
  });
}

export function getMinMax(data: tf.Tensor) {
  return {
    min: data.min(),
    max: data.max()
  }
}

export function normalize(data: tf.Tensor) {
  const { min: dataMin, max: dataMax } = getMinMax(data);

  const normalizedData = data.sub(dataMin).div(dataMax.sub(dataMin));

  return normalizedData;
}

export function standardize(data: tf.Tensor, mean: tf.Tensor, variance: tf.Tensor) {
  const standardizedData = data.sub(mean).div(variance.pow(0.5));

  return standardizedData;
}

export function createModel(inputShape: number) {
  const model = tf.sequential();

  model.add(tf.layers.dense({
    inputShape: [inputShape],
    units: inputShape,
  }));

  model.add(tf.layers.dense({ units: 1 }));

  return model;
}

export async function trainModel(
  model: tf.Sequential,
  features: tf.Tensor,
  labels: tf.Tensor,
  batchSize: number,
  epochs: number
) {
  model.compile({
    optimizer: tf.train.adam(0.1),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse']
  });

  return await model.fit(features, labels, {
    batchSize,
    epochs
  });
}

export function testModel(model: tf.Sequential, testFeatures: tf.Tensor, testLabels: tf.Tensor) {
  const preds = model.predict(testFeatures) as tf.Tensor;

  const SSResidual = testLabels.sub(preds).pow(2).sum().bufferSync().get();

  const SSTotal = testLabels.sub(testLabels.mean()).pow(2).sum().bufferSync().get();

  return 1 - SSResidual / SSTotal;
}

export async function buildAndTestModel(
  features: tf.Tensor,
  labels: tf.Tensor,
  testFeatures: tf.Tensor,
  testLabels: tf.Tensor,
  normalizeData: boolean,
  batchSize: number,
  epochs: number
) {
  const { mean: featuresMean, variance: featuresVariance } = tf.moments(features);
  const stdNormFeatures = normalizeData ? normalize(features) : standardize(features, featuresMean, featuresVariance);

  const model = createModel(features.shape[1]!);

  await trainModel(model, stdNormFeatures, labels, batchSize, epochs);

  const stdNormTestFeatures = normalizeData ? normalize(testFeatures) : standardize(testFeatures, featuresMean, featuresVariance);

  const accuracy = testModel(model, stdNormTestFeatures, testLabels);

  return { accuracy, model, featuresMean, featuresVariance };
}

export async function buildBestModel(
  features: tf.Tensor,
  labels: tf.Tensor,
  testFeatures: tf.Tensor,
  testLabels: tf.Tensor,
  batchLimit: number,
  epochLimit: number
) {
  const bestNormalized = {
    accuracy: 0,
    model: tf.sequential(),
    batchSize: -1,
    epochs: 0,
  };
  
  const bestStandardized = {
    accuracy: 0,
    model: tf.sequential(),
    batchSize: -1,
    epochs: 0,
  };
  
  for (let batchSize = 1; batchSize < batchLimit; batchSize++) {
    for (let numbEpochs = 1; numbEpochs < epochLimit; numbEpochs++) {
      const { accuracy: tempAccuracyNormalized, model: tempModelNormalized } = await buildAndTestModel(features, labels, testFeatures, testLabels, true, batchSize, numbEpochs);
      const { accuracy: tempAccuracyStandardized, model: tempModelStandardized } = await buildAndTestModel(features, labels, testFeatures, testLabels, false, batchSize, numbEpochs);
  
      if (tempAccuracyNormalized > bestNormalized.accuracy && tempAccuracyNormalized <= 1) {
        bestNormalized.accuracy = tempAccuracyNormalized;
        bestNormalized.batchSize = batchSize;
        bestNormalized.epochs = numbEpochs;
        bestNormalized.model = tempModelNormalized;
      }
  
      if (tempAccuracyStandardized > bestStandardized.accuracy && tempAccuracyStandardized <= 1) {
        bestStandardized.accuracy = tempAccuracyStandardized;
        bestStandardized.batchSize = batchSize;
        bestStandardized.epochs = numbEpochs;
        bestStandardized.model = tempModelStandardized;
      }
    }
  }
  
  const bestModel = bestStandardized.accuracy > bestNormalized.accuracy
    ? bestStandardized.model
    : bestNormalized.model;
  
  return bestModel;
}

export function predict(model: tf.Sequential, features: tf.Tensor, mean?: tf.Tensor, variance?: tf.Tensor) {
  const stdNormFeatures = !mean && !variance ? normalize(features) : standardize(features, mean!, variance!);
  return model.predict(stdNormFeatures);
}
