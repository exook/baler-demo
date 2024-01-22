# Introduction
Baler is a tool used to test the feasibility of compressing different types of scientific data using machine learning-based autoencoders. Baler provides you with an easy way to:
1. Train a machine learning model on your data
2. Compress your data with that model, and save the model
3. Decompress the file using the model at a later time

Source code repository: https://github.com/baler-collaboration/baler

# Baler Demo
Baler is a framework, utilizing PyTorch, available as a pip package. This demo provides an example of a script for running baler, which will involve using the package to train, and compress, whilst our script will take care of saving trained models and compressed files to disc. The machine learning models used in baler will be provided from an external directory, baler-models.

Start by cloning the demo repository:
```console
git clone https://github.com/baler-collaboration/baler-demo.git
```
```console
cd baler-demo
```
Then, into this directory, clone the baler-models repository which contains our demo model:
```console
git clone https://github.com/baler-collaboration/baler-models.git
```
Install the Baler framework via pip, at this stage I am certain that there will be a lot of dependencies left to install:
```console
pip install baler-compressor
```
Lastly run the demo script
```console
python3 run.py all
```