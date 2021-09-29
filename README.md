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

To use MLProvCodeGen after installation, open the JupyterLab command line by pressing ctrl+shift+c and enter 
```bash
Code Generation from Provenance Data".
```

![MLProvCodeGen_CommandLine](https://user-images.githubusercontent.com/85288390/135293768-380ba9d1-338a-4d18-96bb-b35a11fb70a7.PNG)

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

### Uninstall

```bash
pip uninstall MLProvCodeGen
```

