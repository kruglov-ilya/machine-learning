import "@tensorflow/tfjs-backend-webgl";

import * as tf from "@tensorflow/tfjs-core";
import * as tfData from "@tensorflow/tfjs-data";
import * as tfLayers from "@tensorflow/tfjs-layers";
import * as tfVis from "@tensorflow/tfjs-vis"
import { FMnistData } from "./fashion-data";


class Neuro {
  canvas?: HTMLCanvasElement;
  rawImage?: HTMLImageElement;
  ctx?: CanvasRenderingContext2D;
  saveButton?: HTMLElement;
  clearButton?: HTMLElement;
  model?: tfLayers.Sequential;

  pos = { x: 0, y: 0 };

  constructor() {
    document.addEventListener('DOMContentLoaded', () => this.run());
  }

  getModel() {

    // In the space below create a convolutional neural network that can classify the 
    // images of articles of clothing in the Fashion MNIST dataset. Your convolutional
    // neural network should only use the following layers: conv2d, maxPooling2d,
    // flatten, and dense. Since the Fashion MNIST has 10 classes, your output layer
    // should have 10 units and a softmax activation function. You are free to use as
    // many layers, filters, and neurons as you like.  
    // HINT: Take a look at the MNIST example.
    this.model = tfLayers.sequential();

    // YOUR CODE HERE
    this.model.add(tfLayers.layers.conv2d({ inputShape: [28, 28, 1], kernelSize: 3, filters: 8, activation: 'relu' }));
    this.model.add(tfLayers.layers.maxPooling2d({ poolSize: [2, 2] }));
    this.model.add(tfLayers.layers.conv2d({ filters: 16, kernelSize: 3, activation: 'relu' }));
    this.model.add(tfLayers.layers.maxPooling2d({ poolSize: [2, 2] }));
    this.model.add(tfLayers.layers.flatten());
    this.model.add(tfLayers.layers.dense({ units: 128, activation: 'relu' }));
    this.model.add(tfLayers.layers.dense({ units: 10, activation: 'softmax' }));

    // Compile the model using the categoricalCrossentropy loss,
    // the tf.train.adam() optimizer, and accuracy for your metrics.
    this.model.compile({ optimizer: tf.train.adam(), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    return this.model;
  }

  async train(model: tfLayers.Sequential, data: FMnistData) {

    // Set the following metrics for the callback: 'loss', 'val_loss', 'accuracy', 'val_accuracy'.
    const metrics = ['loss', 'val_loss', 'accuracy', 'val_accuracy'];// YOUR CODE HERE    


    // Create the container for the callback. Set the name to 'Model Training' and 
    // use a height of 1000px for the styles. 
    const container = { name: 'Model Training', styles: { height: '1000px' } };// YOUR CODE HERE   


    // Use tfvis.show.fitCallbacks() to setup the callbacks. 
    // Use the container and metrics defined above as the parameters.
    const fitCallbacks = tfVis.show.fitCallbacks(container, metrics);// YOUR CODE HERE

    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 6000;
    const TEST_DATA_SIZE = 1000;

    // Get the training batches and resize them. Remember to put your code
    // inside a tf.tidy() clause to clean up all the intermediate tensors.
    // HINT: Take a look at the MNIST example.
    const [trainXs, trainYs] = tf.tidy(() => {
      const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
      return [
        d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
        d.labels
      ];
    });


    // Get the testing batches and resize them. Remember to put your code
    // inside a tf.tidy() clause to clean up all the intermediate tensors.
    // HINT: Take a look at the MNIST example.
    const [testXs, testYs] = tf.tidy(() => {
      const d = data.nextTestBatch(TEST_DATA_SIZE);
      return [
        d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
        d.labels
      ];
    });


    return model.fit(trainXs, trainYs, {
      batchSize: BATCH_SIZE,
      validationData: [testXs, testYs],
      epochs: 10,
      shuffle: true,
      callbacks: fitCallbacks
    });
  }

  setPosition(e: MouseEvent) {
    this.pos.x = e.clientX - 100;
    this.pos.y = e.clientY - 100;
  }

  draw(e: MouseEvent) {
    if (e.buttons != 1) return;
    this.ctx!.beginPath();
    this.ctx!.lineWidth = 24;
    this.ctx!.lineCap = 'round';
    this.ctx!.strokeStyle = 'white';
    this.ctx!.moveTo(this.pos.x, this.pos.y);
    this.setPosition(e);
    this.ctx!.lineTo(this.pos.x, this.pos.y);
    this.ctx!.stroke();
    this.rawImage!.src = this.canvas!.toDataURL('image/png');
  }

  erase() {
    this.ctx!.fillStyle = "black";
    this.ctx!.fillRect(0, 0, 280, 280);
  }

  save() {
    const raw = tf.browser.fromPixels(this.rawImage!, 1);
    const resized = tf.image.resizeBilinear(raw, [28, 28]);
    const tensor = resized.expandDims(0);

    const prediction = this.model!.predict(tensor) as tf.Tensor<tf.Rank>;
    const pIndex = tf.argMax(prediction, 1).dataSync()[0];

    const classNames = ["T-shirt/top", "Trouser", "Pullover",
      "Dress", "Coat", "Sandal", "Shirt",
      "Sneaker", "Bag", "Ankle boot"];


    alert(classNames[pIndex]);
  }

  init() {
    this.canvas = document.getElementById('canvas') as HTMLCanvasElement;
    this.rawImage = document.getElementById('canvasimg')! as HTMLImageElement;
    this.ctx = this.canvas!.getContext("2d")!;
    this.ctx.fillStyle = "black";
    this.ctx.fillRect(0, 0, 280, 280);
    this.canvas.addEventListener("mousemove", e => this.draw(e));
    this.canvas.addEventListener("mousedown", e => this.setPosition(e));
    this.canvas.addEventListener("mouseenter", e => this.setPosition(e));
    this.saveButton = document.getElementById('sb')!;
    this.saveButton.addEventListener("click", () => this.save());
    this.clearButton = document.getElementById('cb')!;
    this.clearButton.addEventListener("click", () => this.erase());
  }


  async run() {
    const data = new FMnistData();
    await data.load();
    const model = this.getModel();
    tfVis.show.modelSummary({ name: 'Model Architecture' }, model);
    await this.train(model, data);
    await model.save('downloads://my_model');
    this.init();
    alert("Training is done, try classifying your drawings!");
  }
}

var neuro = new Neuro()