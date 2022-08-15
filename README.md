# MLProvCodeGen - Machine Learning Provenance Code Generator

![Github Actions Status](https://github.com/fusion-jena/MLProvCodeGen/workflows/Build/badge.svg)[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fusion-jena/MLProvCodeGen/main?urlpath=lab)

MLProvCodeGen is a tool for provenance based code generation of ML experiments in the Jupyter environment. This tool is developed to help ML practitioners and data scientists and can also be used in education to showcase results of ML workflows for a problem with different parameters. It can be used to tune hyperparameters and see the difference in the results.


This extension is composed of a Python package named `MLProvCodeGen`
for the server extension and a NPM package named `MLProvCodeGen`
for the frontend extension.


## Requirements

* JupyterLab >= 3.0

## Install

```bash
pip install MLProvCodeGen
```
## Instructions

To use MLProvCodeGen after installation, open the JupyterLab command line by pressing `ctrl+shift+c` and enter the command
`Code Generation from Provenance Data`

![MLProvCodeGen_CommandLine](https://user-images.githubusercontent.com/85288390/135293768-380ba9d1-338a-4d18-96bb-b35a11fb70a7.PNG)

Here is an example interface:

![MLProvCodeGen_MCC_inputs](https://user-images.githubusercontent.com/85288390/135294673-c435f433-011e-488a-8222-0f53d7c39469.PNG)

And here is an example of a generated notebook:

![NotebookExample_Multiclass_MLProvCodeGen](https://user-images.githubusercontent.com/85288390/135294765-5abdda78-efe7-4549-b0bb-aa91099f1351.PNG)


## Troubleshoot

If you are seeing the frontend extension, but it is not working, check
that the server extension is enabled:

```bash
jupyter server extension list
```

If the server extension is installed and enabled, but you are not seeing
the frontend extension, check the frontend extension is installed:

```bash
jupyter labextension list
```


## Contributing

### Development install

Note: You will need NodeJS to build the extension package.

The `jlpm` command is JupyterLab's pinned version of
[yarn](https://yarnpkg.com/) that is installed with JupyterLab. You may use
`yarn` or `npm` in lieu of `jlpm` below.

```bash
# Clone the repo to your local environment
# Change directory to the MLProvCodeGen directory
# Install package in development mode
pip install -e .
# Link your development version of the extension with JupyterLab
jupyter labextension develop . --overwrite
# Rebuild extension Typescript source after making changes
jlpm run build
```

You can watch the source directory and run JupyterLab at the same time in different terminals to watch for changes in the extension's source and automatically rebuild the extension.

```bash
# Watch the source directory in one terminal, automatically rebuilding when needed
jlpm run watch
# Run JupyterLab in another terminal
jupyter lab
```

With the watch command running, every saved change will immediately be built locally and available in your running JupyterLab. Refresh JupyterLab to load the change in your browser (you may need to wait several seconds for the extension to be rebuilt).

By default, the `jlpm run build` command generates the source maps for this extension to make it easier to debug using the browser dev tools. To also generate source maps for the JupyterLab core extensions, you can run the following command:

```bash
jupyter lab build --minimize=False
```
### Adding new ML experiments

The following steps must be taken to add a new ML experiment to this extension:

1.	Have an existing Python script for your machine learning experiment.
2.	Paste the code into a Jupyter notebook and split it into cells following the execution order of your experiment.
3.	Create a Jinja template for each cell and wrap if-statements around the Python code depending on which variables are important. Refer to existing modules for what the provenance data of your experiment might look like.
4.	Load the templates in a Python procedure that also creates a new notebook element and write their rendered outputs to the notebook.
5.	Expect every local variable for the procedure to be extracted from a dictionary input.
6.	Add HTML input elements to the user interface based on your provenance variables.
7.	Combine the variable values into a JavaScript/TypeScript dictionary.
8.	Create a new server request for your module and pass the dictionary through it as “stringified” JSON data.
9.	Once the frontend, backend, and server connection work, your module has been added successfully.

Note that while these steps might seem complicated, most of them only require copy-pasting already existing code. The only new part for most users is templating through Jinja. However, Jinja has good documentation, and its syntax is very simple, requiring only if-loops.



### Uninstall

```bash
pip uninstall MLProvCodeGen
```


Welcome! 
Completing this survey will take 20-30 Minutes.

Note: Please complete this survey on a computer or laptop.



In this survey, we will ask you to evaluate your experience with MLProvCodeGen and to complete user tasks. 

MLProvCodeGen can be used on a virtual machine, meaning that there will be no changes to your system and no installation required, available at:

 https://mybinder.org/v2/gh/fusion-jena/MLProvCodeGen/HEAD
(Please open the link now as the virtual machine can have varying startup times)

Note: In case you encounter any error messages while following the above link, please try reloading the page or waiting a few minutes. There are several errors that might occur, however most of them can be solved by reloading and/or waiting.



Before we get into the questions, we will provide you with an introduction to MLProvCodeGen:


MLProvCodeGen is an abbreviation for 'Machine Learning Provenance Code Generator'.

Our goal in this research was to find out, whether provenance data can be used to support the end-to-end reproducibility of machine learning experiments.

In short, provenance data is data that contains information about a specific datapoint; how, when, and by whom it was conceived, and by which processes (functions, methods) it was generated.



The functionalities of MLProvCodeGen can be split into 2 parts:

MLProvCodeGen's original purpose was to automatically generate code for training machine learning (ML) models, providing users multiple different options for machine learning tasks, datasets, model parameters, training parameters and evaluation metrics. 
We then extended MLProvCodeGen to generate code according to real-world provenance data models, to automatically capture provenance data from the generated experiments, and to take provenance data files that were captured with MLProvCodeGen as input to generate one-to-one reproductions of the original experiments.
MLProvCodeGen can also generate relational graphs of the captured provenance data, allowing for visual representation of the implemented experiments.



The specific use-cases for this project are twofold: 

1. Image Classification
- We can generate code to train a ML model on image input files to classify handwritten digits (MNIST),

clothing articles (FashionMNIST), and a mix of vehicles and animals (CIFAR10).

2. Multiclass Classification
- We can generate code to train a ML model on tabular data (.csv) to classify different species of iris flowers

and to also test different models using 'toy datasets' which are fake datasets specifically designed to mimic patterns that could occur in real-world data such as spirals.




