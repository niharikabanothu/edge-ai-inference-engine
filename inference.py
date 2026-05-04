import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import time

# Load TFLite model
def load_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Get input/output details
def get_details(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details, output_details

# Preprocess image
def preprocess(image, input_shape):
    image = cv2.resize(image, (input_shape[1], input_shape[2]))
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32) / 255.0
    return image

# Run inference
def run_inference(interpreter, input_details, output_details, image):
    interpreter.set_tensor(input_details[0]['index'], image)
    
    start = time.time()
    interpreter.invoke()
    end = time.time()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    latency = (end - start) * 1000
    return output, latency

# Main
def main():
    MODEL_PATH = "models/model.tflite"
    IMAGE_PATH = "test_image.jpg"
    
    print("Loading model...")
    interpreter = load_model(MODEL_PATH)
    input_details, output_details = get_details(interpreter)
    
    print("Loading image...")
    image = cv2.imread(IMAGE_PATH)
    input_shape = input_details[0]['shape']
    processed = preprocess(image, input_shape)
    
    print("Running inference...")
    output, latency = run_inference(interpreter, input_details, output_details, processed)
    
    print(f"Inference Latency: {latency:.2f} ms")
    print(f"Output: {output}")
    predicted_class = np.argmax(output)
    print(f"Predicted Class: {predicted_class}")

if __name__ == "__main__":
    main()
