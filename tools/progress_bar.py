#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Hikari Software
# X-Endurance

import sys

class ProgressBar(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    # 显示进度条
    self.draw_progress_bar(epoch + 1, EPOCHS)

  def draw_progress_bar(self, cur, total, bar_len=50):
    cur_len = int(cur / total * bar_len)
    sys.stdout.write("\r")
    sys.stdout.write("[{:<{}}] {}/{}".format("=" * cur_len, bar_len, cur, total))
    sys.stdout.flush()
