import tflite_runtime.interpreter as tflite
interpreter = tflite.Interpreter(model_path="mobilenet_v1_1.0_224_quant.tflite")
interpreter.allocate_tensors()
print("✅ TFLite works!")