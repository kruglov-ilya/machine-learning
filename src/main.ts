import '@tensorflow/tfjs-backend-webgl';

import * as tf from "@tensorflow/tfjs-core";
import * as tfData from "@tensorflow/tfjs-data";
import * as tfLayers from "@tensorflow/tfjs-layers"

const csvUrl =
  "https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/boston-housing-train.csv";

interface DataType extends tf.TensorContainerObject {
  xs: tf.Tensor;
  ys: tf.Tensor;
}

async function run() {
  // We want to predict the column "medv", which represents a median value of a
  // home (in $1000s), so we mark it as a label.
  const csvDataset = tfData.csv(csvUrl, {
    columnConfigs: {
      medv: {
        isLabel: true,
      },
    },
  }) as any as tfData.Dataset<DataType> & tfData.CSVDataset;

  debugger;
  console.log(csvDataset);

  debugger;
  // Number of features is the number of column names minus one for the label
  // column.
  const numOfFeatures = (await csvDataset.columnNames()).length - 1;

  // Prepare the Dataset for training.
  const flattenedDataset = csvDataset
    .map(({ xs, ys }) => {
      // Convert xs(features) and ys(labels) from object form (keyed by column
      // name) to array form.
      return { xs: Object.values(xs), ys: Object.values(ys) };
    })
    .batch(10);

  console.log(csvDataset);

  console.log(flattenedDataset);

  // Define the model.
  const model = tfLayers.sequential();
  model.add(
    tfLayers.layers.dense({
      inputShape: [numOfFeatures],
      units: 1,
    })
  );
  model.compile({
    optimizer: tf.train.sgd(0.000001),
    loss: "meanSquaredError",
  });

  // Fit the model using the prepared Dataset
  return model.fitDataset(flattenedDataset, {
    epochs: 10,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(epoch, logs?.loss);
      },
    },
  });
}

run().then(() => console.log("Done"));
