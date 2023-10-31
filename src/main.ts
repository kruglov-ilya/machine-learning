import * as tf from "@tensorflow/tfjs-core";
import * as posenet from "@tensorflow-models/posenet";
import "@tensorflow/tfjs-backend-webgl";

const videoElement: HTMLVideoElement = document.getElementById(
  "video"
) as HTMLVideoElement;
const canvas: HTMLCanvasElement = document.getElementById(
  "output"
) as HTMLCanvasElement;
const ctx: CanvasRenderingContext2D = canvas.getContext("2d")!;

async function setupCamera(): Promise<void> {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      "Browser API navigator.mediaDevices.getUserMedia not available"
    );
  }

  videoElement.width = 600;
  videoElement.height = 500;

  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  videoElement.srcObject = stream;

  return new Promise((resolve) => {
    videoElement.onloadedmetadata = () => {
      canvas.width = videoElement.width;
      canvas.height = videoElement.height;
      resolve();
    };
  });
}

async function loadPoseNet(): Promise<posenet.PoseNet> {
  const net = await posenet.load();
  return net;
}

function drawKeypoints(
  keypoints: posenet.Keypoint[],
  minConfidence: number,
  context: CanvasRenderingContext2D
) {
  for (const keypoint of keypoints) {
    if (keypoint.score >= minConfidence) {
      const { y, x } = keypoint.position;
      context.beginPath();
      context.arc(x, y, 10, 0, 2 * Math.PI);
      context.fillStyle = "aqua";
      context.fill();
    }
  }
}

function drawSkeleton(
  keypoints: posenet.Keypoint[],
  minConfidence: number,
  context: CanvasRenderingContext2D
) {
  const adjacentKeyPoints = posenet.getAdjacentKeyPoints(
    keypoints,
    minConfidence
  );

  adjacentKeyPoints.forEach((keypoints) => {
    context.beginPath();
    context.moveTo(keypoints[0].position.x, keypoints[0].position.y);
    context.lineTo(keypoints[1].position.x, keypoints[1].position.y);
    context.lineWidth = 2;
    context.strokeStyle = "aqua";
    context.stroke();
  });
}

async function bindPage() {
  await tf.setBackend("webgl"); // Set the backend to WebGL
  await tf.ready(); // Wait for the backend to be ready

  await setupCamera();
  videoElement.play();

  const net = await loadPoseNet();
  console.log("PoseNet model loaded.");

  detectPoseInRealTime(videoElement, net);
}

function detectPoseInRealTime(video: HTMLVideoElement, net: posenet.PoseNet) {
  async function poseDetectionFrame() {
    const poses = await net.estimatePoses(video, {
      flipHorizontal: false,
      decodingMethod: "single-person",
    });

    ctx.clearRect(0, 0, video.width, video.height);
    ctx.save();
    ctx.drawImage(video, 0, 0, video.width, video.height);
    ctx.restore();

    poses.forEach(({ keypoints }) => {
      drawKeypoints(keypoints, 0.5, ctx);
      drawSkeleton(keypoints, 0.5, ctx);
    });

    requestAnimationFrame(poseDetectionFrame);
  }

  poseDetectionFrame();
}

window.onload = bindPage;
