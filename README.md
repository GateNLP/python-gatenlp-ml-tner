# python-gatenlp-ml-tner

Token classification training and application using transformers via the tner package.

See: https://github.com/asahi417/tner / https://pypi.org/project/tner/

IMPORTANT: for this package, a modified version of tner is required (see installation instructions below)!

### Installation:

* For now this package does NOT require the packages 
  it depends on in order to avoid dependency hell. See below 
  for how to install the required packages.
* create a new environment (or activate a gatenlp environment you already have instead)
  (e.g. `conda create -y -n gatenlp-tner python=3.7`) 
  and activate it
* install the PyTorch version compatible with your machine 
  see [PyTorch Installation](https://pytorch.org/get-started/locally/)
* install the modified version of TNER:
    * `python -m pip install -U git+https://github.com/johann-petrak/tner-modified.git`
* if not using a gatenlp environment, install gatenlp, e.g.: `python -m pip install -U gatenlp[all]`

### Usage

See [example notebook](examples/train-conll2003)

