import fs from 'fs';
import * as tf from '@tensorflow/tfjs-node';
import shuffleSeed from 'shuffle-seed';
import Papa from 'papaparse';
export function getKnnData(file) {
    const { data, errors, meta } = Papa.parse(fs.readFileSync(file, { encoding: 'utf8' }), {
        header: true,
        dynamicTyping: true,
        transform: function (value, field) {
            if (field !== 'passedemissions')
                return value;
            return value === 'TRUE' ? 1 : 0;
        }
    });
    if (errors.length > 0) {
        throw new Error(`Error parsing file: ${file}`);
    }
    return data;
}
export function convertFeaturesToTensor(data, featureColumns, labelColumns, testSplit) {
    return tf.tidy(() => {
        // tf.util.shuffle(data);
        data = shuffleSeed.shuffle(data, 'phrase');
        const features = data.map(d => featureColumns.map(feature => d[feature]));
        const labels = data.map(d => labelColumns.map(label => d[label]));
        const featureTensor = tf.tensor(features);
        return {
            features: featureTensor.slice(testSplit),
            labels: labels.slice(testSplit),
            testFeatures: featureTensor.slice(0, testSplit),
            testLabels: labels.slice(0, testSplit),
        };
    });
}
