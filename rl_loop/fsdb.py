# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Filesystem DB: poor man's worker coordination strategy.

This module works with local filesystems.
"""
import os
import sys
sys.path.insert(0, '.')

from absl import flags
import tensorflow as tf
import re

gfile = tf.io.gfile

from rl_loop import shipname

flags.DEFINE_string(
    'base_dir', None,
    'Root directory if using local FS as the database.')

FLAGS = flags.FLAGS

class FSDB:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def _path(self, *args):
        return os.path.join(self.base_dir, *args)

    def working_dir(self):
        return self._path('work_dir')

    def models_dir(self):
        return self._path('models')

    def selfplay_dir(self):
        return self._path('data', 'selfplay')

    def holdout_dir(self):
        return self._path('data', 'holdout')

    def sgf_dir(self):
        return self._path('sgf')

    def eval_dir(self):
        return self._path('sgf', 'eval')

    def golden_chunk_dir(self):
        return self._path('data', 'golden_chunks')

    def flags_path(self):
        return self._path('flags.txt')

    def eval_flags_path(self):
        return self._path('eval-flags.txt')

    def get_pbs(self):
        all_pbs = gfile.Glob(os.path.join(self.models_dir(), '*.pb'))
        return all_pbs

    def get_models(self):
        """Finds all models, returning a list of model number and names
        sorted increasing.

        Returns: [(13, 000013-modelname), (17, 000017-modelname), ...etc]
        """
        all_models = gfile.Glob(os.path.join(self.models_dir(), '*.meta'))
        model_filenames = [os.path.basename(m) for m in all_models]
        model_numbers_names = sorted([
            (shipname.detect_model_num(m), shipname.detect_model_name(m))
            for m in model_filenames])
        return model_numbers_names

    def get_latest_model(self):
        """Finds the latest model, returning its model number and name

        Returns: (17, 000017-modelname)
        """
        return self.get_models()[-1]

    def get_latest_pb(self):
        pbs = self.get_pbs()
        if not pbs:
            return None
        pb = os.path.basename(pbs[-1])
        return shipname.detect_model_num(pb), pb

    def get_model(self, model_num):
        """Given a model number 17, returns its full name 000017-modelname."""
        model_names_by_num = dict(self.get_models())
        return model_names_by_num[model_num]

    def get_hour_dirs(self, root=None):
        """Gets the directories under selfplay_dir that match YYYY-MM-DD-HH."""
        root = root or self.selfplay_dir()
        if not gfile.Exists(root):
            return []
        return list(filter(lambda s: re.match(r"\d{4}-\d{2}-\d{2}-\d{2}", s),
                           gfile.ListDirectory(root)))

    def get_games(self, model_name):
        return gfile.Glob(os.path.join(self.selfplay_dir(), model_name, '*.zz'))

    def game_counts(self, n_back=20):
        """Prints statistics for the most recent n_back models"""
        for _, model_name in self.get_models()[-n_back:]:
            games = self.get_games(model_name)
            print("Model: {}, Games: {}".format(model_name, len(games)))

# For backward compatibility and ease of use with singleton pattern if desired
_instance = None

def _get_instance():
    global _instance
    if _instance is None:
        if FLAGS.base_dir is None:
            raise ValueError("base_dir must be set!")
        _instance = FSDB(FLAGS.base_dir)
    return _instance

def working_dir(): return _get_instance().working_dir()
def models_dir(): return _get_instance().models_dir()
def selfplay_dir(): return _get_instance().selfplay_dir()
def holdout_dir(): return _get_instance().holdout_dir()
def sgf_dir(): return _get_instance().sgf_dir()
def eval_dir(): return _get_instance().eval_dir()
def golden_chunk_dir(): return _get_instance().golden_chunk_dir()
def flags_path(): return _get_instance().flags_path()
def eval_flags_path(): return _get_instance().eval_flags_path()
def get_pbs(): return _get_instance().get_pbs()
def get_models(): return _get_instance().get_models()
def get_latest_model(): return _get_instance().get_latest_model()
def get_latest_pb(): return _get_instance().get_latest_pb()
def get_model(model_num): return _get_instance().get_model(model_num)
def get_hour_dirs(root=None): return _get_instance().get_hour_dirs(root)
def get_games(model_name): return _get_instance().get_games(model_name)
def game_counts(n_back=20): return _get_instance().game_counts(n_back)
