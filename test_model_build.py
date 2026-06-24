import sys
import tensorflow as tf
from absl import flags
import dual_net
import go

def test_model_build():
    # Initialize flags with default values
    FLAGS = flags.FLAGS
    FLAGS(sys.argv[:1])  # Parse with no arguments to use defaults

    # Assuming AGZ features which has 17 planes
    model = dual_net.get_model()

    # Create dummy input: (batch_size, height, width, channels)
    # 19x19 board, 17 planes
    dummy_input = tf.random.uniform((1, go.N, go.N, 17))

    # Run forward pass
    policy, value = model(dummy_input, training=False)

    print(f"Policy shape: {policy.shape}")
    print(f"Value shape: {value.shape}")

    # Assertions
    # 19*19 + 1 = 361 + 1 = 362
    assert policy.shape == (1, 362), f"Expected policy shape (1, 362), got {policy.shape}"
    assert value.shape == (1, 1), f"Expected value shape (1, 1), got {value.shape}"

    print("Model build and forward pass verification successful!")

if __name__ == "__main__":
    test_model_build()
