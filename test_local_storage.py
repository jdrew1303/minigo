import os
import tempfile
import unittest
from rl_loop.fsdb import FSDB

class TestLocalStorage(unittest.TestCase):
    def test_fsdb_sgf_storage(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fsdb = FSDB(tmp_dir)

            # Create a model-specific SGF directory
            model_name = "000001-test"
            sgf_dir = os.path.join(fsdb.sgf_dir(), model_name)
            os.makedirs(sgf_dir)

            # Define dummy SGF content
            sgf_content = "(;GM[1]FF[4]SZ[19]KM[7.5];B[pd];W[dp])"
            sgf_filename = "test_game.sgf"
            sgf_path = os.path.join(sgf_dir, sgf_filename)

            # Save the dummy SGF file
            with open(sgf_path, "w") as f:
                f.write(sgf_content)

            # Verify the file exists
            self.assertTrue(os.path.exists(sgf_path))

            # Read back and verify content
            with open(sgf_path, "r") as f:
                read_content = f.read()

            self.assertEqual(sgf_content, read_content)
            print(f"Successfully wrote and read SGF to {sgf_path}")

if __name__ == "__main__":
    unittest.main()
