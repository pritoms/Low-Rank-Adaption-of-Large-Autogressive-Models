#!/usr/bin/env bash

mkdir -p data/webtext/
cd data/webtext/
wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/language-modeling/data/webtext/train.txt
wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/language-modeling/data/webtext/valid.txt
wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/language-modeling/data/webtext/test.txt
