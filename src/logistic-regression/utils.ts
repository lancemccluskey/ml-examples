import Papa from 'papaparse';
import fs from 'fs';
import * as tf from '@tensorflow/tfjs-node';
import shuffleSeed from 'shuffle-seed';

export function getLogRegData(file: string) {
  const { data, errors, meta } = Papa.parse(
    fs.readFileSync(file, { encoding: 'utf8' }), {
      header: true,
      dynamicTyping: true,
      transform: function (value, field) {
        if (field !== 'passedemissions') return value;

        return value === 'TRUE' ? 1 : 0;
      }
  });

  if (errors.length > 0) {
    throw new Error(`Error parsing file: ${file}`);
  }

  return data;
}

export function splitData(data: any[], featureColumns: string[], labelColumns: string[], testSplit: number) {
    data = shuffleSeed.shuffle(data, 'phrase');

    const features = data.map(d => featureColumns.map(feature => d[feature]));
    const labels = data.map(d => labelColumns.map(label => d[label] as number));

    return {
      features: features.slice(testSplit),
      labels: labels.slice(testSplit),
      testFeatures: features.slice(0, testSplit),
      testLabels: labels.slice(0, testSplit),
    }
}
