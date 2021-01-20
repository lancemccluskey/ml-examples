import { tensor, zeros, moments, ones } from '@tensorflow/tfjs-node';
export class LogisticRegression {
    constructor(features, labels, options) {
        this.costHistory = [];
        this.mean = undefined;
        this.variance = undefined;
        this.labels = tensor(labels);
        this.features = this.processFeatures(features);
        this.options = Object.assign({
            learningRate: 0.1,
            epochs: 1000,
            decisionBoundary: 0.5,
            batchSize: 32
        }, options);
        this.weights = zeros([this.features.shape[1], 1]);
    }
    processFeatures(features) {
        const featuresTensor = tensor(features);
        const standardizedFeatures = (this.mean && this.variance)
            ? featuresTensor.sub(this.mean).div(this.variance.pow(0.5))
            : this.standardize(featuresTensor);
        // We concat with ones because of the b term, remember mx + b
        return ones([standardizedFeatures.shape[0], 1]).concat(standardizedFeatures, 1);
    }
    standardize(features) {
        const { mean, variance } = moments(features, 0);
        this.mean = mean;
        this.variance = variance;
        return features.sub(mean).div(variance.pow(0.5));
    }
    train() {
        const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize);
        for (let epoch = 0; epoch < this.options.epochs; epoch++) {
            for (let batch = 0; batch < batchQuantity; batch++) {
                const { batchSize } = this.options;
                const startIndex = batch * batchSize;
                const featureSlice = this.features.slice([startIndex, 0], // aka row index
                [batchSize, -1] // number of rows to take, -1 means all columns
                );
                const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);
                this.gradientDescent(featureSlice, labelSlice);
            }
            this.recordCost();
            this.updateLearningRate();
        }
    }
    gradientDescent(features, labels) {
        // Derivative of Cross Entropy
        // Matrix Multiply Features * Weights
        const currentGuesses = features.matMul(this.weights).sigmoid();
        // Subtract guesses from labels
        const differences = currentGuesses.sub(labels);
        // Multiply by transpose of features tensor
        const slopes = features
            .transpose()
            .matMul(differences)
            .div(features.shape[0]);
        // Update the weights
        this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
    }
    recordCost() {
        // Calculate Cross Entropy
        // -1 / N * (Actual * log(Guesses) + (1 - Actual) * log(1 - Guesses))
        // Guesses is sigmoid of mx + b
        const guesses = this.features.matMul(this.weights).sigmoid();
        // Actual * log(Guesses)
        const leftTerm = this.labels.transpose().matMul(guesses.log());
        // (1 - Actual) * log(1 - Guesses)
        // 1 - Actual === -1 * Actual + 1
        const rightTerm = this.labels
            .mul(-1)
            .add(1)
            .transpose()
            .matMul(guesses.mul(-1).add(1).log());
        // -1 / N * (leftTerm + rightTerm)
        const cost = leftTerm.add(rightTerm)
            .div(this.features.shape[0])
            .mul(-1)
            .bufferSync()
            .get(0, 0);
        this.costHistory.unshift(cost);
    }
    updateLearningRate() {
        if (this.costHistory.length < 2) {
            return;
        }
        this.options.learningRate = this.costHistory[0] > this.costHistory[1]
            ? this.options.learningRate /= 2
            : this.options.learningRate *= 1.05;
    }
    predict(observations) {
        return this.processFeatures(observations)
            .matMul(this.weights)
            .sigmoid()
            .greater(this.options.decisionBoundary) // returns boolean
            .cast('float32');
    }
    test(testFeatures, testLabels) {
        const predictions = this.predict(testFeatures);
        const testLabelsTensor = tensor(testLabels);
        const incorrect = predictions
            .sub(testLabelsTensor)
            .abs()
            .sum()
            .bufferSync()
            .get();
        // (TotalPredictions - Incorrect) / TotalPredictions gives us % accuracy
        return (predictions.shape[0] - incorrect) / predictions.shape[0];
    }
}
