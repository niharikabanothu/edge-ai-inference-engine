import tensorflow as tf
import numpy as np

# Load and convert model to TFLite with quantization
def quantize_model(saved_model_path, output_path):
    print("Loading model...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    
    # Apply quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    print("Applying quantization...")
    tflite_model = converter.convert()
    
    # Save quantized model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Quantized model saved to {output_path}")
    return tflite_model

# Compare model sizes
def compare_sizes(original_path, quantized_path):
    import os
    original_size = os.path.getsize(original_path) / (1024 * 1024)
    quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
    reduction = ((original_size - quantized_size) / original_size) * 100
    
    print(f"Original Model Size: {original_size:.2f} MB")
    print(f"Quantized Model Size: {quantized_size:.2f} MB")
    print(f"Size Reduction: {reduction:.2f}%")

# Main
def main():
    SAVED_MODEL_PATH = "models/saved_model"
    QUANTIZED_PATH = "models/model.tflite"
    
    quantize_model(SAVED_MODEL_PATH, QUANTIZED_PATH)
    print("Quantization complete!")
    print("Model ready for Raspberry Pi deployment!")

if __name__ == "__main__":
    main()
