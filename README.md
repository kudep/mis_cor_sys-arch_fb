mis_cor_sys-arch_fb-tensorflow
===

Multi-layer Recurrent Neural Networks (LSTM, RNN) for system for correcting typos in Python using Tensorflow.

Inspired from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).

## Requirements
- [Tensorflow 1.0](http://www.tensorflow.org)

## Basic Usage
To train with default parameters on the tinyshakespeare corpus, run `python train.py`. To access all the parameters use `python train.py --help`.

To sample from a checkpointed model, `python sample.py`.

## Tensorboard
To visualize training progress, model graphs, and internal state histograms:  fire up Tensorboard and point it at your `log_dir`.  E.g.:
```bash
$ tensorboard --logdir=./logs/
```

Then open a browser to [http://localhost:6006](http://localhost:6006) or the correct IP/Port specified.


## Roadmap
- [ ] Add explanatory comments
- [ ] Expose more command-line arguments
- [ ] Compare accuracy and performance with char-rnn
- [ ] More Tensorboard instrumentation

## Contributing
Please feel free to:
* Leave feedback in the issues
* Open a Pull Request