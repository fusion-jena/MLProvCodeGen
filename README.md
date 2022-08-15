# MLProvCodeGen - Machine Learning Provenance Code Generator

![Github Actions Status](https://github.com/fusion-jena/MLProvCodeGen/workflows/Build/badge.svg)[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fusion-jena/MLProvCodeGen/main?urlpath=lab)

## Install

```bash
pip install MLProvCodeGen
```

Our goal in this research was to find out, whether provenance data can be used to support the end-to-end reproducibility of machine learning experiments.

In short, provenance data is data that contains information about a specific datapoint; how, when, and by whom it was conceived, and by which processes (functions, methods) it was generated.

![provenance data example](https://user-images.githubusercontent.com/85288390/184615649-925cc96b-9372-4e27-90eb-1fe2e20c2f98.PNG)

The functionalities of MLProvCodeGen can be split into 2 parts:

MLProvCodeGen's original purpose was to automatically generate code for training machine learning (ML) models, providing users multiple different options for machine learning tasks, datasets, model parameters, training parameters and evaluation metrics. 
We then extended MLProvCodeGen to generate code according to real-world provenance data models, to automatically capture provenance data from the generated experiments, and to take provenance data files that were captured with MLProvCodeGen as input to generate one-to-one reproductions of the original experiments.
MLProvCodeGen can also generate relational graphs of the captured provenance data, allowing for visual representation of the implemented experiments.



The specific use-cases for this project are twofold: 

1. Image Classification
- We can generate code to train a ML model on image input files to classify handwritten digits (MNIST),

clothing articles (FashionMNIST), and a mix of vehicles and animals (CIFAR10).

![MNIST example](https://user-images.githubusercontent.com/85288390/184615694-2ca7f720-3a8a-4775-8ed3-921fabc1294b.PNG)

2. Multiclass Classification
- We can generate code to train a ML model on tabular data (.csv) to classify different species of iris flowers

![iris example](https://user-images.githubusercontent.com/85288390/184615789-0f307120-43ff-4bc1-b95f-67efe139fa1e.PNG)

and to also test different models using 'toy datasets' which are fake datasets specifically designed to mimic patterns that could occur in real-world data such as spirals.


![Spiral example](https://user-images.githubusercontent.com/85288390/184615851-3f19081c-0b30-4c42-b314-41caa72f7f53.PNG)

# How to use MLProvCodeGen

Please open MLProvCodeGen by using the **Binder Button** at the top of this page. This opens a virtual installation.

The JupyterLab interface should look like this: 

![jupyterlab startup](https://user-images.githubusercontent.com/85288390/184616379-53cf9ff3-8026-4a7a-9b0b-2d9aa48f1bfa.png)

Please proceed by pressing the 'MLProvCodeGen' button located in the 'other' section to open the extension.

![MLProvCodeGen startup](https://user-images.githubusercontent.com/85288390/184616409-7550e57b-23b2-4016-9390-b6b1fabda61d.png)

Here is an example interface:

![MLProvCodeGen_MCC_inputs](https://user-images.githubusercontent.com/85288390/135294673-c435f433-011e-488a-8222-0f53d7c39469.PNG)

And generated notebooks look like this: 

![execute notebook button red](https://user-images.githubusercontent.com/85288390/184616631-98c853b5-3fac-40ec-9652-4295b735858c.png)

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
