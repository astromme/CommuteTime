#!/bin/sh

DIR=$(dirname $0)

virtualenv "$DIR/virtualenv"
source "$DIR/virtualenv/bin/activate"
pip install ujson
pip install matplotlib
pip install numpy 
pip install scipy
pip install scikit-learn