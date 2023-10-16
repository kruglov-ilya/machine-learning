import "@tensorflow/tfjs-backend-webgl";

import * as tf from "@tensorflow/tfjs-core";
import * as tfData from "@tensorflow/tfjs-data";
import * as tfLayers from "@tensorflow/tfjs-layers";

const trainingUrl = "/wdbc-train.csv";
const testingUrl = "/wdbc-test.csv";

interface DataType extends tf.TensorContainerObject {
  xs: tf.Tensor;
  ys: tf.Tensor;
}

type NormalCSVDataset = tfData.Dataset<DataType> & tfData.CSVDataset;

async function run() {
  const trainingData = tfData.csv(trainingUrl, {
    columnConfigs: {
      diagnosis: {
        isLabel: true,
      },
    },
  }) as any as NormalCSVDataset;

  console.log(await trainingData.toArray());

  const convertedTrainingData = trainingData
    .map(({ xs, ys }) => {
      return { xs: Object.values(xs), ys: Object.values(ys) };
    })
    .batch(10);

  const testingData = tfData.csv(testingUrl, {
    columnConfigs: {
      diagnosis: {
        isLabel: true,
      },
    },
  }) as any as NormalCSVDataset;

  console.log(await testingData.toArray());

  const convertedTestingData = testingData
    .map(({ xs, ys }) => {
      return { xs: Object.values(xs), ys: Object.values(ys) };
    })
    .batch(10);

  const numOfFeatures = (await trainingData.columnNames()).length - 1;

  // Define the model.
  const model = tfLayers.sequential();
  model.add(
    tfLayers.layers.dense({
      inputShape: [numOfFeatures],
      units: 1,
    })
  );
  model.compile({
    optimizer: tf.train.adam(0.06),
    loss: "categoricalCrossentropy",
  });

  await model.fitDataset(convertedTrainingData, {
    epochs: 100,
    validationData: convertedTestingData,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(
          "Epoch: " + epoch + " Loss: " + logs?.loss + " Accuracy: " + logs?.acc
        );
      },
    },
  });
  await model.save("downloads://my_model");
}

run().then(() => console.log("Done"));
