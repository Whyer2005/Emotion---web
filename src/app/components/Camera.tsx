"use client";
import { useEffect, useRef, useState } from "react";
import { InferenceSession, Tensor } from "onnxruntime-web";

export default function Camera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [session, setSession] = useState<InferenceSession | null>(null);
  const [classes, setClasses] = useState<string[]>([]);
  const [emotion, setEmotion] = useState<string>("");

  useEffect(() => {
    async function loadModel() {
      const s = await InferenceSession.create("/web_models/emotion_yolo11n_cls.onnx");
      setSession(s);

      const res = await fetch("/web_models/class.json");
      const data = await res.json();
      setClasses(data);
    }
    loadModel();

    async function startCamera() {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    }
    startCamera();
  }, []);

  async function captureAndPredict() {
    if (!videoRef.current || !session || classes.length === 0) return;

    const canvas = document.createElement("canvas");
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext("2d");
    ctx?.drawImage(videoRef.current, 0, 0, 64, 64);

    const imageData = ctx?.getImageData(0, 0, 64, 64);
    if (!imageData) return;

    const rgbData = new Float32Array(3 * 64 * 64);
    for (let i = 0; i < 64 * 64; i++) {
      rgbData[i] = imageData.data[i * 4] / 255;             // R
      rgbData[i + 64 * 64] = imageData.data[i * 4 + 1] / 255; // G
      rgbData[i + 2 * 64 * 64] = imageData.data[i * 4 + 2] / 255; // B
    }

    const input = new Tensor("float32", rgbData, [1, 3, 64, 64]);

    const output = await session.run({ images: input });
    const logits = output[Object.keys(output)[0]] as Tensor;
    const arr = logits.data as Float32Array;

    const predIdx = arr.indexOf(Math.max(...arr));
    setEmotion(classes[predIdx]);
  }

  // 🔄 เรียก captureAndPredict ทุก 2 วินาที
  useEffect(() => {
    const interval = setInterval(() => {
      captureAndPredict();
    }, 2000); // 2000 ms = 2 วินาที

    return () => clearInterval(interval);
  }, [session, classes]);

  return (
    <div className="flex flex-col items-center gap-4">
      <video ref={videoRef} autoPlay playsInline className="w-[400px] h-[300px] border" />
      {emotion && (
        <p className="text-xl font-semibold text-blue-600 dark:text-yellow-300">
          อารมณ์ที่ตรวจพบ: {emotion}
        </p>
      )}
    </div>
  );
}
