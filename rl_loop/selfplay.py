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
import os
import sys
import time
from absl import app, flags
sys.path.insert(0, '.')

from rl_loop import fsdb
import mask_flags

# "_nr" signifies "No Resign", aka calibration game, which will use a different
# set of flags and will not update its flags from a remote flagfile.
flags.DEFINE_enum('mode', None, ['cc'],
                  'Which setup to use: cc on GPU.')

FLAGS = flags.FLAGS


def run_cc():
    _, model_name = fsdb.get_latest_model()
    num_games_finished = len(fsdb.get_games(model_name))
    if num_games_finished > 25000:
        print("{} has enough games! ({})".format(
            model_name, num_games_finished))
        time.sleep(10 * 60)
        sys.exit()

    mask_flags.checked_run([
        'bazel-bin/cc/selfplay',
        '--model=tf,{}'.format(model_name),
        '--mode=selfplay',
        '--output_dir={}/{}'.format(
            fsdb.selfplay_dir(), model_name),
        '--holdout_dir={}/{}'.format(
            fsdb.holdout_dir(), model_name),
        '--sgf_dir={}/{}'.format(
            fsdb.sgf_dir(), model_name),
        '--flagfile=rl_loop/distributed_flags'])


def main(unused_argv):
    flags.mark_flags_as_required(['mode'])
    if FLAGS.mode == 'cc':
        run_cc()


if __name__ == '__main__':
    app.run(main)
