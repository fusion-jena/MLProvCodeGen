import nbformat as nbf
import json
import sys
from jinja2 import Environment, FileSystemLoader
import webbrowser

def app(code):
    nb = nbf.v4.new_notebook()

    text = """\
    # My first automatic Jupyter Notebook
    This is an auto-generated notebook."""

    code2 = """\
    %pylab inline
    hist(normal(size=2000), bins=50);"""

    nb['cells'] = [nbf.v4.new_markdown_cell(text),
                    nbf.v4.new_code_cell(code2)]


    cellText = str(code)
    nb['cells'].append(nbf.v4.new_markdown_cell(cellText))



    nbf.write(nb, 'GeneratedNotebooks/userInputNotebook.ipynb')

    return code

def IC_pytorch(user_inputs):
    nb = nbf.v4.new_notebook()

    text = """
# Image Classification

Building a machine learning model to solve Image Classification using the PyTorch framework.<br>
Image Classification is one of the basic pattern recognition exercises. <br>
Using Image files as its input, a model trained for Image classification will split a set of images into a given number of classes. <br>
<br>
This Notebook has been generated automatically using the JupyterLab extension ***MLProvCodeGen***.
<br>
The original Source Code is from this application https://github.com/jrieke/traingenerator <br>
Made by: https://www.jrieke.com/ Twitter: https://twitter.com/jrieke
"""
    nb['cells'] = [nbf.v4.new_markdown_cell(text)]

    file_loader = FileSystemLoader('jinjaTemplates/IC_pytorch')
    env = Environment(loader=file_loader, trim_blocks=True, lstrip_blocks=True)
# GET VARIABLES
    #TODOvisualization_tool = user_inputs['visualization_tool']['tool']
    visualization_tool = 'notAtAll'
    notebook = True
    data_format = user_inputs['entity']['ex:Data Ingestion Data']['ex:data_format']
    dataset = user_inputs['entity']['ex:Data Ingestion Data']['ex:dataset_id']
    num_classes = user_inputs['entity']['ex:Data Ingestion Data']['ex:classes']['$']
    checkpoint = user_inputs['entity']['ex:Model Parameters Data']['ex:save_checkpoint']['$']
    lr = user_inputs['entity']['ex:Model Parameters Data']['ex:optimizer_learning_rate']['$']
    gpu = user_inputs['entity']['ex:Model Parameters Data']['ex:gpu_enable']['$']
    model_func = user_inputs['entity']['ex:Model Parameters Data']['ex:model_name']
    pretrained = user_inputs['entity']['ex:Model Parameters Data']['ex:pretrained']['$']
    loss = user_inputs['entity']['ex:Model Parameters Data']['ex:loss_function']
    optimizer = user_inputs['entity']['ex:Model Parameters Data']['ex:optimizer']
    batch_size = user_inputs['entity']['ex:Training Data']['ex:batch_size']['$']
    num_epochs = user_inputs['entity']['ex:Training Data']['ex:epochs']['$']
    print_every = user_inputs['entity']['ex:Training Data']['ex:print_progress']['$']
    seed = user_inputs['entity']['ex:Training Data']['ex:seed']['$']
    
#-----------------------------------------------------------------------------
    # installs
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Installs
Install required packages before running"""))
    template = env.get_template('001_installs.jinja')
    output = template.render(visualization_tool = visualization_tool)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    # imports
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Imports"""))
    template = env.get_template('002_imports.jinja')
    output = template.render(visualization_tool = visualization_tool, data_format = data_format, checkpoint = checkpoint)
    nb['cells'].append(nbf.v4.new_code_cell(output))
    
    # init provenance
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Provenance Data"""))
    template = env.get_template('003_provenanceInit.jinja')
    output = template.render(visualization_tool = visualization_tool)
    nb['cells'].append(nbf.v4.new_code_cell(output))
    
    #Data Ingestion
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Data Ingestion"""))
    template = env.get_template('004_dataIngestion.jinja')
    output = template.render(data_format = data_format, dataset = dataset, pretrained = pretrained, visualization_tool = visualization_tool, checkpoint = checkpoint, num_classes = num_classes)
    nb['cells'].append(nbf.v4.new_code_cell(output))
    
    #Data preparation
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Data Preparation"""))
    template = env.get_template('005_dataPreparation.jinja')
    output = template.render(data_format = data_format, dataset = dataset, pretrained = pretrained, gpu = gpu)
    nb['cells'].append(nbf.v4.new_code_cell(output))
    
    #Data Segregation
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Data Segregation"""))
    template = env.get_template('006_dataSegregation.jinja')
    output = template.render(data_format = data_format, dataset = dataset, pretrained = pretrained, gpu = gpu, batch_size = batch_size, print_every=print_every, num_classes = num_classes)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #Model
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Model"""))
    template = env.get_template('007_model.jinja')
    output = template.render(model_func = model_func, pretrained = pretrained, num_classes=num_classes, loss = loss, optimizer = optimizer, visualization_tool = visualization_tool, lr = lr, gpu = gpu, checkpoint = checkpoint)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #Training
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Training"""))
    template = env.get_template('008_training.jinja')
    output = template.render(visualization_tool = visualization_tool, lr = lr, num_epochs = num_epochs, data_format = data_format, print_every = print_every, seed=seed)
    nb['cells'].append(nbf.v4.new_code_cell(output))
    
    #Evaluation
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Evaluation"""))
    template = env.get_template('009_evaluation.jinja')
    output = template.render(visualization_tool = visualization_tool, lr = lr, num_epochs = num_epochs, data_format = data_format)
    nb['cells'].append(nbf.v4.new_code_cell(output))
    
    # Generate Provenance Data
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Generate Provenance Data"""))
    template = env.get_template('010_generateProvenanceData.jinja')
    output = template.render()
    nb['cells'].append(nbf.v4.new_code_cell(output))

    # Write Provenance Data
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Write Provenance Data"""))
    template = env.get_template('011_writeProvenanceData.jinja')
    output = template.render()
    nb['cells'].append(nbf.v4.new_code_cell(output))
    
    # Open Provenance Data
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Open Provenance Data"""))
    template = env.get_template('012_openProvenanceData.jinja')
    output = template.render()
    nb['cells'].append(nbf.v4.new_code_cell(output))

    nbf.write(nb, 'GeneratedNotebooks/ImageClassification_PyTorch.ipynb')
    reply = {"greetings": "success"}
    return reply


def MulticlassClassification(user_inputs):
    #_(sys.executable, "cd", "..")
    nb = nbf.v4.new_notebook()
    
    text = """
# Multiclass Classification

Building a neural network to solve a multiclass classification exercise using the PyTorch framework.<br>
Classification is one of the basic machine learning exercises. A trained model aims to predict the class of an input unit with high accuracy. <br>
This neural network uses supervised learning, meaning that the input datasets also provide target labels to train the model with. <br>
<br>
This Notebook has been generated automatically using the JupyterLab extension ***MLProvCodeGen***.
<br>

Original Source Code and inspiration from this article https://janakiev.com/blog/pytorch-iris/ <br>
Original author: N. Janakiev https://github.com/njanakiev Twitter: https://twitter.com/njanakiev
        """

    nb['cells'] = [nbf.v4.new_markdown_cell(text)]

    file_loader = FileSystemLoader('jinjaTemplates/MulticlassClassification')
    env = Environment(loader=file_loader, trim_blocks=True, lstrip_blocks=True)
    dataset = user_inputs['entity']['ex:Data Ingestion Data']['ex:dataset_id']
    random_seed = user_inputs['entity']['ex:Data Segregation Data']['ex:random_state']['$']
    test_split = user_inputs['entity']['ex:Data Segregation Data']['ex:test_size']['$']
    activation_func = user_inputs['entity']['ex:Model Parameters Data']['ex:activation_function']
    neuron_number = user_inputs['entity']['ex:Model Parameters Data']['ex:neuron_number']['$']
    optimizer = user_inputs['entity']['ex:Model Parameters Data']['ex:optimizer']
    loss_func = user_inputs['entity']['ex:Model Parameters Data']['ex:loss_function']
    epochs = user_inputs['entity']['ex:Training Data']['ex:epochs']['$']
    #lr can be either 'NULL' or an actual value, therefore exception required
    try:
        lr = user_inputs['entity']['ex:Model Parameters Data']['ex:optimizer_learning_rate'][0]
    except KeyError:
        lr = user_inputs['entity']['ex:Model Parameters Data']['ex:optimizer_learning_rate']['$']

    use_gpu = user_inputs['entity']['ex:Model Parameters Data']['ex:gpu_enable']['$']
    default = user_inputs['entity']['ex:Model Parameters Data']['ex:optimizer_default_learning_rate']['$']


    #installs
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Installs
Install required packages before running"""))
    template = env.get_template('001_installs.jinja')
    output = template.render()
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #imports
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Imports"""))
    template = env.get_template('002_imports.jinja')
    output = template.render(dataset = dataset)
    nb['cells'].append(nbf.v4.new_code_cell(output))
    
    #provenance (experiment, hardware, packages)
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Provenance Data """))
    template = env.get_template('003_provenanceInit.jinja')
    output = template.render()
    nb['cells'].append(nbf.v4.new_code_cell(output))
    
    #dataIngestion
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Data Ingestion"""))
    template = env.get_template('004_dataIngestion.jinja')
    output = template.render(dataset = dataset)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #datapreparation
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Data Preparation"""))
    template = env.get_template('005_dataPreparation.jinja')
    output = template.render()
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #dataSegregation
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Data Segregation"""))
    template = env.get_template('006_dataSegregation.jinja')
    output = template.render(random_seed = random_seed, test_split = test_split)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #dataVisualization
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Data Visualization"""))
    template = env.get_template('007_dataVisualization.jinja')
    output = template.render(dataset = dataset)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #model
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Model"""))
    template = env.get_template('008_model.jinja')
    output = template.render(optimizer = optimizer,default = default, lr = lr, loss_func = loss_func, activation_func = activation_func, use_gpu = use_gpu, dataset = dataset, neuron_number = neuron_number)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #training
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Training"""))
    template = env.get_template('009_training.jinja')
    output = template.render(epochs = epochs, dataset = dataset)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #evaluation
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Evaluation"""))
    template = env.get_template('010_evaluation.jinja')
    output = template.render()
    nb['cells'].append(nbf.v4.new_code_cell(output))
    
    #confusion matrix
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Confusion Matrix"""))
    template = env.get_template('011_confusionMatrix.jinja')
    output = template.render()
    nb['cells'].append(nbf.v4.new_code_cell(output))
    
    #F1 score
    nb['cells'].append(nbf.v4.new_markdown_cell("""### F1 Score"""))
    template = env.get_template('012_F1.jinja')
    output = template.render()
    nb['cells'].append(nbf.v4.new_code_cell(output))
    
    #mean absolute error & mean squared error
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Mean Absolute Error & Mean Squared Error"""))
    template = env.get_template('013_meanErrors.jinja')
    output = template.render()
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #ROC
    nb['cells'].append(nbf.v4.new_markdown_cell("""### ROC"""))
    template = env.get_template('014_ROC.jinja')
    output = template.render()
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #Generate Provenance Data
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Generate Provenance Data"""))
    template = env.get_template('015_generateProvenanceData.jinja')
    output = template.render()
    nb['cells'].append(nbf.v4.new_code_cell(output))
    
    #Write Provenance Data
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Write Provenance Data"""))
    template = env.get_template('016_writeProvenanceData.jinja')
    output = template.render()
    nb['cells'].append(nbf.v4.new_code_cell(output))
    
    #Open Provenance Data Button
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Show Provenance Data"""))
    template = env.get_template('017_openProvenanceData.jinja')
    output = template.render()
    nb['cells'].append(nbf.v4.new_code_cell(output))

    nbf.write(nb, 'GeneratedNotebooks/MulticlassClassification.ipynb')

    reply = {"greetings": "success"}
    return reply
    
def openNotebook(user_inputs):
    
    reply ={"greetings": "success"}
    return reply