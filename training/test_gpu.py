import tensorflow as tf

print("\nTensorFlow Version:")
print(tf.__version__)

print("\nGPU Available:")
print(tf.config.list_physical_devices('GPU'))