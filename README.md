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
