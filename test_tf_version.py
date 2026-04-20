import tensorflow as tf
print(f'TF Version: {tf.__version__}')
assert tf.__version__.startswith('2.'), f'TensorFlow version must be 2.x, but got {tf.__version__}'
