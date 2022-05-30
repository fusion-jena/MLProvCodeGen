import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { ICommandPalette, MainAreaWidget } from '@jupyterlab/apputils';
import { Widget } from '@lumino/widgets';
import { ISettingRegistry } from '@jupyterlab/settingregistry';
import { requestAPI } from './handler';
import { ILauncher } from '@jupyterlab/launcher';

/* eslint-disable no-useless-escape */

async function generateNotebook(requestname: string, objBody: object, content: Widget) {
	try {
		const reply = await requestAPI<any>(
			requestname,
			{
			body: JSON.stringify(objBody),
			method: 'POST'
			}
		);
		console.log(reply)
		return reply
// ------------------------------------------------------------------------------------------------------------------------------- //
    } catch (reason) {
		console.error(
			`Error on POST /extension/`+requestname+` ${objBody}.\n${reason}`
		);
    } 
}

async function activate (app: JupyterFrontEnd, palette: ICommandPalette, launcher: ILauncher, settingRegistry: ISettingRegistry | null) {
	console.log('JupyterLab extension extension is activated!');
// setup and HTTP Request test; used to check if the server extension is enabled locally/ ob Binder
    if (settingRegistry) {
      settingRegistry
        .load(plugin.id)
        .then(settings => {
          console.log('extension settings loaded:', settings.composite);
        })
        .catch(reason => {
          console.error('Failed to load settings for extension.', reason);
        });
    }

    requestAPI<any>('get_example')
      .then(data => {
        console.log(data);
      })
      .catch(reason => {
        console.error(
          `The extension server extension appears to be missing.\n${reason}`
        );
      });
	  
	const dataToSend = { 'name': 'MLProvCodeGen1.0' };  
	try {
		const reply = await requestAPI<any>('post_example', {
		body: JSON.stringify(dataToSend),
		method: 'POST'
		});
		console.log(reply);
	} catch (reason) {
		console.error(
		`ERROR on post_example ${dataToSend}.\n${reason}`
		);
	}
// ------------------------------------------------------------------------------------------------------------------------------- //
  // Create a blank content widget inside of a MainAreaWidget
  const content = new Widget();
  const widget = new MainAreaWidget({ content });
  widget.id = 'MLProvCodeGen-jupyterlab';
  content.id = 'MLProvCodeGen-content' // id used to add scrollbars in base.css
  widget.title.label = 'MLProvCodeGen';
  widget.title.closable = true;
// ------------------------------------------------------------------------------------------------------------------------------- //
	// Header
	const headerFlex = document.createElement('div');
	content.node.appendChild(headerFlex);
	headerFlex.id = 'headerFlex'
	headerFlex.innerHTML = `
		<div class="flex-container">
			<div><h2>MLProvCodeGen</h2></div>
			<div><h3>Generate machine learning scripts using provenance data</h3></div>
			<div>Input a MLProvCodeGen provenance data file <b><u>or</u></b></div>
			<div>Use the input elements below.</div>
			<div>Hover over input elements for explanations.</div>
		</div>
	`
// ------------------------------------------------------------------------------------------------------------------------------- //  
  // Button to reset widget
  const resetFlex = document.createElement('div');
  content.node.appendChild(resetFlex);
  resetFlex.id = 'resetFlex'
  resetFlex.innerHTML = `
		<div class="flex-container">
			<div><button id="reset" type="Button"> Reset this tab </button></div>
		</div>
		`;

  resetFlex.addEventListener('click', event => {
    const nodeList = content.node.childNodes;
    console.log(nodeList);
    while (nodeList.length > 0) {
      nodeList[0].remove();
    }
	content.node.appendChild(headerFlex);
	content.node.appendChild(resetFlex);
	content.node.appendChild(provInputFlex);
	content.node.appendChild(problemSelectionFlex);
	content.node.appendChild(problemSelectionButton);
  });
// ------------------------------------------------------------------------------------------------------------------------------- //
	const provInputFlex = document.createElement('div');
	content.node.appendChild(provInputFlex);
	provInputFlex.id = 'provInputFlex';
	provInputFlex.innerHTML = `
		<div class="flex-container2">
			<div><b>Insert a MLProvCodeGen Provenance File:</b></div>
			<div><input type="file"</div>
			<div><button id="provenanceSubmit" type="button"> Submit Provenance File </button>  </div>
		</div>
	`
	
	provInputFlex.addEventListener('change', event => {
		let file = (<HTMLInputElement>event.target).files![0];
		document.getElementById('provenanceSubmit')!.addEventListener('click', async event => {
			console.log(file);
			
			let reader = new FileReader();
			reader.readAsText(file)
			reader.onload = async function() {
				//console.log(reader.result);
				var provenanceDataObj = JSON.parse(reader.result!.toString());
				console.log(provenanceDataObj); 
				//console.log(provenanceDataObj.entity.experiment_info['experimentinfo:task_type']); 
				const taskName = provenanceDataObj.entity['ex:Experiment Info Data']['ex:task_type'];
				var path = window.location.href + '/tree/GeneratedNotebooks/'
				var notebookPath = "('"+path+taskName+".ipynb', 'MLProvCodeGen')";
				console.log('path:' +path);
				//var notebookPath = "('http://localhost:8888/lab/tree/extension/GeneratedNotebooks/"+provenanceDataObj.entity.experiment_info['experimentinfo:task_type']+".ipynb', 'MLProvCodeGen')";
				var openCall = `onclick="window.open`+notebookPath+`">`;
				console.log(openCall);
				
				const reply = await generateNotebook(taskName, provenanceDataObj, content)
				console.log(reply)
				if (reply['greetings'] === 'success') {
					const successInput = document.createElement('div');
					content.node.insertBefore(successInput, problemSelectionFlex)
					successInput.innerHTML = `
						<div class="flex-container3">
							<div><p>Your code has been generated successfully.</div>
							<div><button id="openButtonRight" type="button" `+openCall+` Open Notebook </button> </div>
						</div>
					`
				}
			};
		}); // end of submitProvenanceFile event listener	
	}); // end of provenanceInput event listener
// ------------------------------------------------------------------------------------------------------------------------------- //
  const problemSelectionFlex = document.createElement('div');
  content.node.appendChild(problemSelectionFlex);
  problemSelectionFlex.id = "problemSelectionFlex"
  problemSelectionFlex.innerHTML = `
		<div class="flex-container2">
			<div><b>Submit data through input elements:</b></div>
			<div>
				<label for="exercise">Choose a machine learning exercise:</label>
				<select name="exercise" id="exercise">
					<option value="ImageClassification"> Image Classification </option>
					<option value="MulticlassClassification"> Multiclass Classification</option>
				</select>
			</div>
		</div>
  `
  // ------------------------------------------------------------------------------------------------------------------------------- //
  // submit button for problem selection
  const problemSelectionButton = document.createElement('div');
  content.node.appendChild(problemSelectionButton);
  problemSelectionButton.id='problemSelectionButton';
  problemSelectionButton.innerHTML = `
		<div class="flex-container2">
			<div><button id="inputButton" type="button"> Submit </button></div>
		</div>
		`;

  problemSelectionButton.addEventListener('click', event => {
    const problemSubmit = (<HTMLSelectElement>(
      document.getElementById('exercise')
    )).value;
	
switch (problemSubmit) {
    case 'ImageClassification':
        const IC_dataFormat = document.createElement('div');
		content.node.appendChild(IC_dataFormat);
		IC_dataFormat.innerHTML = `
					<div class="flex-container2">
						<div><b><u> Data Ingestion</u></b></div>
						<div title="Select the data format.">
							<label for="data">Which data format do you want to use?</label>
							<select name="data" id="data">
								<option value="Public dataset"> Public dataset </option>
							</select>
						</div>
					</div>
						`;
						//<option value="Numpy arrays"> Numpy arrays </option>
						//<option value="Image files"> Image files </option>
		const IC_dataset = document.createElement('div');
		content.node.appendChild(IC_dataset);
		IC_dataset.innerHTML = `
					<div class="flex-container2">
						<div title="The 'Fake data set' consists of 110 randomly generated images.\nMNIST is a database of 70000 image files. The images contain handwritten digits.\nFashionMNIST is a dataset of 70000 image files. The images contain clothing articles split into 10 classes.\nCIFAR10 is a dataset of 60000 image files. The images contain 10 classes of vehicles and animals.">
						<label for="dataSelection">Select your dataset:</label>
						<select name="dataSelection" id="dataSelection">
							<option value="FakeData"> Fake Data for Evaluation </option>
							<option value="MNIST"> MNIST </option>
							<option value="FashionMNIST"> FashionMNIST </option>
							<option value="CIFAR10"> CIFAR10 </option>
							<option value="user"> Use your own data by adding it to the notebook later </option>
						</select>
						</div>	
					</div>
						`;
						
		const IC_classes = document.createElement('div');
        content.node.appendChild(IC_classes);
		IC_classes.innerHTML = `
					<div class="flex-container2">
						<div title="Number of classes that the dataset has.\nDefault for public datasets is 10 classes.">
							<label for="quantity">How many classes/output units?</label>
							<input type="number" id="quantity" name="quantity" value="10">
						</div>
					</div>
				`;
						
		const IC_preprocessing_text = document.createElement('div');
        content.node.appendChild(IC_preprocessing_text);
        IC_preprocessing_text.innerHTML = `
					<div class="flex-container2">
						<div><b><u> Data Preparation</u></b></div>
						<div title="Preprocessing changes the input data to fit our model.">
							<text><i>preprocessing: Resize(256), CenterCrop(224), ToTensor(), grayscale to RGB</i></text>
						</div>
					</div>
					`;

		const IC_segregation_text = document.createElement('div');
        content.node.appendChild(IC_segregation_text);
        IC_segregation_text.innerHTML = `
					<div class="flex-container2">
						<div><b><u> Data Segregation</u></b></div>
						<div title="Data Segregation splits the available data into training data and testing data.\nMNIST and FashionMNIST consist of 60000 training images (10 Classes with 6000 samples for each class) and 10000 testing examples.\nCIFAR10 consists of 50000 training images and 10000 testing images.\nThe fake dataset generates 100 training examples and 10 testing examples.">
							<label><i>Public datasets use premade testing datasets.</i></label>
						</div>
					</div>
					`;
		
		const seed = document.createElement('div');
		content.node.appendChild(seed);
		seed.innerHTML = `
					<div class="flex-container2">
						<div title="Seeds determine the sequence of numbers that a pseudorandom number generator generates.\nWhen the same seed is used for data segregation on the same dataset multiple times, then the training and testing datasets will always be identical.">
							<label for="seed"> Random Seed</label>
							<input type="number" id="seed" name="seed" value="2">
						</div>
					</div>
						`;
						
		const IC_useGPU = document.createElement('div');
		content.node.appendChild(IC_useGPU);
		IC_useGPU.innerHTML = `
					<div class="flex-containerReverse">
						<div><b><u> Model Parameters</u></b></div>
						<div title="Not all GPUs work with the useCuda() function.!">
							<input type="checkbox" id="useGPU" name="useGPU" value="useGPU" checked>
							<label for="useGPU"> Use GPU if available? </label><br>
						</div>
					</div>
						`;
		
		const IC_model = document.createElement('div');
        content.node.appendChild(IC_model);
        IC_model.innerHTML = `
					<div class="flex-container2">
						<div title="PyTorch allows us to use machine learning models from research papers.\nCheck the documentation for more information & the original publications!">
							<label for="model">Select a model:</label>
							<select name="model" id="model">
								<option value="resnet18"> resnet18 </option>
								<option value="shufflenet_v2_x1_0"> shufflenet v2 </option>
								<option value="vgg16"> vgg16 </option>
							</select>
						</div>
					</div>
						`;
		const IC_pretrained = document.createElement('div');
		content.node.appendChild(IC_pretrained);
		IC_pretrained.innerHTML = `
					<div class="flex-containerReverse">
						<div title="Models can be initialised with pre-computed parameters.\nThis option has no impact on training time.">
							<input type="checkbox" id="preTrainedModel" name="preTrainedModel" value="preTrainedModel">
							<label for="preTrainedModel"> Do you want to use a pre-trained model?</label><br>
						</div>
					</div>
						`;

		const IC_optimizer = document.createElement('div');
		content.node.appendChild(IC_optimizer);
		IC_optimizer.innerHTML = `
					<div class="flex-container2">
						<div title="An optimizer changes the parameters of a model after each batch.\nDifferent exercises might require different optimizers to maximise performance, so try out multiple ones!">
							<label for="optimizer"> Optimizer </label>
							<select name="optimizer" id="optimizer">
								<option value="Adam"> Adam </option>
								<option value="Adadelta"> Adadelta </option>
								<option value="Adagrad"> Adagrad </option>
								<option value="Adamax"> Adamax </option>
								<option value="RMSprop"> RMSprop </option>
								<option value="SGD"> SGD </option>
							</select>
						</div>	
					</div>
				`;
						
		const IC_learningRate = document.createElement('div');
		content.node.appendChild(IC_learningRate);
		IC_learningRate.innerHTML = `
					<div class="flex-container2">
						<div title="The learning rate tells the model how much the model parameters should be changed per batch.">
							<label for="rate"> Learning rate</label>
							<input type="number" id="rate" name="rate" value="0.001">
						</div>
					</div>
				`;
						
		const IC_lossFunction = document.createElement('div');
		content.node.appendChild(IC_lossFunction);
		IC_lossFunction.innerHTML = `
					<div class="flex-container2">
						<div title="Loss functions compute how good a model's predictions are.\nIf a model performs well, then loss will be small.">
							<label for="lossFunc"> Loss function</label>
							<select name="lossFunc" id="lossFunc">
								<option value="CrossEntropyLoss"> CrossEntropyLoss </option>
								<option value="NLLLoss"> NLLLoss </option>
							</select>
						</div>
					</div>
				`;
				
		const IC_epochs = document.createElement('div');
		content.node.appendChild(IC_epochs);
		IC_epochs.innerHTML = `
					<div class="flex-container2">
						<div><b><u> Training</u></b></div>
						<div title="For each epoch, the whole dataset will be iterated over once.\nIncreasing the # of epochs such that the model is trained longer might improve performance.">
							<label for="epochs">How many epochs?</label>
							<input type="number" id="epochs" name="epochs" value="2">
						</div>
					</div>
				`;
				
		const IC_batchSize = document.createElement('div');
		content.node.appendChild(IC_batchSize);
		IC_batchSize.innerHTML = `
					<div class="flex-container2">
						<div title="Batch size defines how much data is input into the model before changing its parameters.\nWe recommend size 10 for FakeData and 128 for real data.">
							<label for="batches"> Batch Size</label>
							<input type="number" id="batches" name="batches" value="10">
						</div>
					</div>
				`;
		 
		const IC_checkpoint = document.createElement('div');
		content.node.appendChild(IC_checkpoint);
		IC_checkpoint.innerHTML = `
					<div class="flex-containerReverse">
						<div title="This option saves your model to local files.">
							<input type="checkbox" id="modelCheckpoint" name="modelCheckpoint" value="modelCheckpoint">
							<label for="modelCheckpoint"> Save model checkpoint after each epoch?</label><br>
						</div>
						<div><i>Alert: This option uses a lot of storage space.</i></div>
					</div>
				`;
						
		const IC_printProgress = document.createElement('div');
		content.node.appendChild(IC_printProgress);
		IC_printProgress.innerHTML = `
					<div class="flex-container2">
						<div title="This option defines how often the console updates during training.">
							<label for="printProgress"> Print progress every ... batches</label>
							<input type="number" id="printProgress" name="printProgress" value="1">
						</div>
					</div>
				`;
		/*const IC_logging = document.createElement('div');
		content.node.appendChild(IC_logging);
		IC_logging.innerHTML = `
					<div class="flex-container2">
						<div title="Logging is currently unavailable.">
							<label for="logs"> How to log metrics </label>
							<select name="logs" id="logs">
								<option value="notAtAll"> Not at all </option>
								<option value="Tensorboard"> Tensorboard </option>
								<option value="Aim"> Aim </option>
								<option value="Weights & Biases"> weightsAndBiases </option>
								<option value="comet.ml"> comet.ml </option>
							</select>
						</div>
					</div>
						`;*/
		const submitButtonIC = document.createElement('div');
		content.node.appendChild(submitButtonIC);
		submitButtonIC.innerHTML = `
					<div class="flex-container2">
						<div><button id="inputButton" type="button"> Submit your values </button></div>  
					</div>
						`;
// ------------------------------------------------------------------------------------------------------------------------------- //
        submitButtonIC.addEventListener('click', async event => {
            const exerciseValue = (<HTMLSelectElement>(
              document.getElementById('exercise')
            )).value;
            const modelValue = (<HTMLSelectElement>(
              document.getElementById('model')
            )).value;
            const dataValue = (<HTMLSelectElement>(
              document.getElementById('data')
            )).value;
            const dataSelectionValue = (<HTMLSelectElement>(
              document.getElementById('dataSelection')
            )).value;
            const lossFuncValue = (<HTMLSelectElement>(
              document.getElementById('lossFunc')
            )).value;
            const optimizerValue = (<HTMLSelectElement>(
              document.getElementById('optimizer')
            )).value;
            /*const logsValue = (<HTMLSelectElement>(
              document.getElementById('logs')
            )).value;*/
            const quantityValue = (<HTMLInputElement>(
              document.getElementById('quantity')
            )).value;
            const rateValue = (<HTMLInputElement>(
              document.getElementById('rate')
            )).value;
            const batchesValue = (<HTMLInputElement>(
              document.getElementById('batches')
            )).value;
            const epochsValue = (<HTMLInputElement>(
              document.getElementById('epochs')
            )).value;
            const printProgressValue = (<HTMLInputElement>(
              document.getElementById('printProgress')
            )).value;
			const seedValue = (<HTMLInputElement>(
              document.getElementById('seed')
            )).value;
            let preTrainedModelValue = 3;
            let useGPUValue = 3;
            let modelCheckpointValue = 3;
            if (
              (<HTMLInputElement>document.getElementById('preTrainedModel'))
                .checked
            ) {
              preTrainedModelValue = 1;
            } else {
              preTrainedModelValue = 0;
            }

            if ((<HTMLInputElement>document.getElementById('useGPU')).checked) {
              useGPUValue = 1;
            } else {
              useGPUValue = 0;
            }

            if (
              (<HTMLInputElement>document.getElementById('modelCheckpoint'))
                .checked
            ) {
              modelCheckpointValue = 1;
            } else {
              modelCheckpointValue = 0;
            }
// ------------------------------------------------------------------------------------------------------------------------------- //
            const objBody = {
				exercise: exerciseValue,
				'entity':{
					'ex:Data Ingestion Data': {
						'ex:data_format': dataValue,
						'ex:dataset_id': dataSelectionValue,
						'ex:classes': {
							'$': quantityValue, 
							'type': typeof(quantityValue),
						},
					},
					'ex:Model Parameters Data': {
						'ex:model_name': modelValue,
						'ex:pretrained': {
							'$': preTrainedModelValue,
							'type': typeof(preTrainedModelValue),
						},
						'ex:gpu_enable': {
							'$': useGPUValue,
							'type': typeof(useGPUValue),
						},
						'ex:save_checkpoint': {
							'$': modelCheckpointValue,
							'type': typeof(modelCheckpointValue),
						},
						'ex:loss_function': lossFuncValue,
						'ex:optimizer': optimizerValue,
						'ex:optimizer_learning_rate': {
							'$': rateValue,
							'type': typeof(rateValue)
						},
					},
					'ex:Training Data': {
						'ex:batch_size': {
							'$': batchesValue,
							'type': typeof(batchesValue),
						},
						'ex:epochs': {
							'$': epochsValue,
							'type': typeof(epochsValue),
						},
						'ex:print_progress': {
							'$': printProgressValue,
							'type': typeof(printProgressValue),
						},
						'ex:seed': {
							'$': seedValue,
							'type': typeof(seedValue)
						}
					},
					'visualization_tool':{
						//'tool' : logsValue,
						'tool' : 'notAtAll'
					},
				}
			};
			var method = 'ImageClassification_pytorch'
			const reply = await generateNotebook(method, objBody, content)
			console.log(reply);
			
			if (reply['greetings'] === 'success') {
				var path = window.location.href + '/tree/GeneratedNotebooks/ImageClassification_PyTorch.ipynb'
				console.log(path)
				const success_message = document.createElement('text');
				content.node.appendChild(success_message);
				success_message.textContent =
				'Your code has been generated successfully. Press the button below to open it.';
				
				const notebook_open = document.createElement('div');
				content.node.appendChild(notebook_open);
				notebook_open.innerHTML = `
								<div class="flex-container2">
									<div><button id="inputButton" type="button" onclick="window.open('`+path+`', 'MLProvCodeGen')"> Open Notebook </button></div>
								</div>
								`;
			}
        }); // end of submitButton event listener
	break;
case 'MulticlassClassification':
      // UI Inputs
        const MC_dataset = document.createElement('div');
        content.node.appendChild(MC_dataset);
        MC_dataset.innerHTML = `
					<div class="flex-container2">
						<div><b><u> Data Ingestion</u></b></div>
						<div title="In data ingestion, you can select a specific dataset. This option selects the raw data, before any preprocessing.">
							<label for="dataset">Which dataset do you want to use?</label>
							<select name="dataset" id="dataset">
								<option value="Spiral"> Spiral </option>
								<option value="Iris"> Iris </option>
								<option value="Aggregation"> Aggregation </option>
								<option value="R15"> R15 </option>
								<option value="User"> Use your own Data (.csv) </option>
							</select>
						</div>
					</div>
						`;
					
		const MC_preprocessing_text = document.createElement('div');
        content.node.appendChild(MC_preprocessing_text);
        MC_preprocessing_text.innerHTML = `
					<div class="flex-container2">
						<div><b><u> Data Preparation</u></b></div>
						<div title="Scale feature range to [0,1]">
							<label> <i>preprocessing: MinMaxScaler</i></label>
						</div>
					</div>
					`;		
				
        const MC_random_seed = document.createElement('div');
        content.node.appendChild(MC_random_seed);
        MC_random_seed.innerHTML = `
					<div class="flex-container2">
						<div><b><u> Data Segregation</u></b></div>
							<div title="Data preparation is the pipeline step that takes the raw data from data ingestion and performs a number of operation on that data to better fit it to the machine learning task at hand and the model that we plan on using.\n The MinMaxScaler scales the data's feature range to [0,1]">
							<label for="random_seed">Random Seed for data Segregation (default: 2)</label>
							<input type="number" id="random_seed" name="random_seed" value="2">
						</div>
					</div>
						`;

        const MC_test_split = document.createElement('div');
        content.node.appendChild(MC_test_split);
        MC_test_split.innerHTML = `
					<div class="flex-container2">
						<div title="The input number describes the % of available datapoints to be used for testing.\nThe default value splits the input data into 20% testing and 80% training data.">
							<label for="test_split">Testing data split (default: 0.2)</label>
							<input type="number" id="test_split" name="test_split" value="0.2">
						</div>
					</div>
						`;

		const MC_use_gpu = document.createElement('div');
        content.node.appendChild(MC_use_gpu);
        MC_use_gpu.innerHTML = `
					<div class="flex-containerReverse">
						<div><b><u> Model Parameters</u></b></div>
						<div title="Depending on the GPU used, training can be sped up significantly by not only using the CPU's, but also the GPU's computational capabilities.">
							<input type="checkbox" id="use_gpu" name="use_gpu" value="use_gpu" checked>
							<label for="use_gpu"> Use GPU if available? </label><br>
						</div>
					</div>
						`;

        const MC_activation_func = document.createElement('div');
        content.node.appendChild(MC_activation_func);
        MC_activation_func.innerHTML = `
					<div class="flex-container2">
						<div title="The activation function of a neural network defines how the output of a neuron is computed.\nIf the accuracy of the generated neural network is unsatisfactory, try changing the activation funciton and compare the results!">
							<label for="activation_func">Activation function:</label>
							<select name="activation_func" id="activation_func">
								<option value="F.softmax(self.layer3(x), dim=1)"> Softmax </option>
								<option value="torch.sigmoid(self.layer3(x))"> Sigmoid </option>
								<option value="torch.tanh(self.layer3(x))"> Tanh </option>
							</select>
						</div>
					</div>
						`;

        const MC_neuron_number = document.createElement('div');
        content.node.appendChild(MC_neuron_number);
        MC_neuron_number.innerHTML = `
					<div class="flex-container2">
						<div title="The generated neural network has 3 layers: input, middle, and output.\nThis options defines the # of neurons for the middle layer.">
							<label for="neuron_number">How many Neurons for middle layer?</label>
							<input type="number" id="neuron_number" name="neuron_number" value="50">
						</div>
						<div><i>(Input and output neurons are separate.)</i></div>
					</div>
						`;

        const MC_optimizer = document.createElement('div');
        content.node.appendChild(MC_optimizer);
        MC_optimizer.innerHTML = `
					<div class="flex-container2">
<div title="An optimizer changes the parameters of a neural network after each batch.\nDifferent exercises might require different optimizers to maximise performance, so try out multiple ones!">
							<label for="optimizer"> Optimizer </label>
							<select name="optimizer" id="optimizer">
								<option value="torch.optim.Adam("> Adam </option>
								<option value="torch.optim.Adadelta("> Adadelta </option>
								<option value="torch.optim.Adagrad("> Adagrad </option>
								<option value="torch.optim.Adamax("> Adamax </option>
								<option value="torch.optim.RMSprop("> RMSprop </option>
								<option value="torch.optim.SGD("> SGD </option>
							</select>
						</div>
					</div>
						`;
        const MC_default_lr = document.createElement('div');
        content.node.appendChild(MC_default_lr);
        MC_default_lr.innerHTML = `
					<div class="flex-containerReverse">
						<div title="The learning rate tells the model how much the model parameters should be changed per batch.">
							<input type="checkbox" id="default" name="default" value="default" checked>
							<label for="default"> Use optimizers default learning rate? </label><br>
						</div>
					</div>
						`;

        const MC_lr = document.createElement('div');
        content.node.appendChild(MC_lr);
        MC_lr.innerHTML = `
					<div class="flex-container2">
						<div title="The learning rate tells the model how much the model parameters should be changed per batch.">
							<label for="rate"> Learning rate</label>
							<input type="number" id="rate" name="rate" value="0.001">
						</div>
					</div>
						`;
        const MC_loss = document.createElement('div');
        content.node.appendChild(MC_loss);
        MC_loss.innerHTML = `
					<div class="flex-container2">
						<div title="Loss functions compute how good a model's predictions are.\nIf a model performs well, then loss will be small.">
							<label for="lossFunc"> Loss function</label>
							<select name="lossFunc" id="lossFunc">
								<option value="nn.CrossEntropyLoss()"> CrossEntropyLoss </option>
								<option value="nn.NLLLoss()"> NLLLoss </option>
								<option value="nn.MultiMarginLoss()"> MultiMarginLoss </option>
							</select>
						</div>
					</div>
						`;
						
		const MC_epochs = document.createElement('div');
        content.node.appendChild(MC_epochs);
        MC_epochs.innerHTML = `
					<div class="flex-container2">
						<div><b><u> Training</u></b></div>
						<div title="For each epoch, the whole dataset will be iterated over once.\nIncreasing the # of epochs such that the model is trained longer might improve performance.">
							<label for="epochs">How many Epochs?</label>
							<input type="number" id="epochs" name="epochs" value="100">
						</div>
					</div>
						`;				

        const MC_submitButton = document.createElement('div');
        content.node.appendChild(MC_submitButton);
        MC_submitButton.innerHTML = `
					<div class="flex-container2">
						<div><button id="inputButton" type="button"> Submit your values </button></div>
					</div>
						`;
        MC_submitButton.addEventListener('click', async event => {
// ------------------------------------------------------------------------------------------------------------------------------- //
			// Get Variables
			const exercise = (<HTMLSelectElement>(
				document.getElementById('exercise')
			)).value;
			const dataset = (<HTMLSelectElement>(
				document.getElementById('dataset')
			)).value;
			const activation_func = (<HTMLSelectElement>(
				document.getElementById('activation_func')
			)).value;
			const optimizer = (<HTMLSelectElement>(
				document.getElementById('optimizer')
			)).value;
			const loss_func = (<HTMLSelectElement>(
				document.getElementById('lossFunc')
			)).value;

			const test_split = (<HTMLInputElement>(
				document.getElementById('test_split')
			)).value;
			const neuron_number = (<HTMLInputElement>(
				document.getElementById('neuron_number')
			)).value;
			const epochs = (<HTMLInputElement>document.getElementById('epochs'))
				.value;
			const lr = (<HTMLInputElement>document.getElementById('rate')).value;
			const random_seed = (<HTMLInputElement>(
				document.getElementById('random_seed')
			)).value;

			let defaultValue, use_gpu;
			if ((<HTMLInputElement>document.getElementById('default')).checked) {
				defaultValue = 1;
			} else {
				defaultValue = 0;
			}

			if ((<HTMLInputElement>document.getElementById('use_gpu')).checked) {
				use_gpu = 1;
			} else {
				use_gpu = 0;
			}
// ------------------------------------------------------------------------------------------------------------------------------- //
			// convert variables into JSON/ input Object
			const objBody = {
				exercise: exercise,
				'entity':{
					'ex:Data Ingestion Data': {
						'ex:dataset_id': dataset
					},			
					'ex:Data Segregation Data': {
						'ex:test_size':{
							'$': test_split,
							'type' : typeof(test_split),
						},
						'ex:random_state': {
							'$': random_seed,
							'type': typeof(random_seed),
						},
					},
					'ex:Model Parameters Data': {
						'ex:gpu_enable': {
							'$':use_gpu,
							'type':typeof(use_gpu),
						},
						'ex:neuron_number': {
							'$':neuron_number,
							'type':typeof(neuron_number),
						},
						'ex:loss_function': loss_func,
						'ex:optimizer': optimizer,
						'ex:optimizer_default_learning_rate':{
							'$':defaultValue,
							'type': typeof(defaultValue),
						},
						'ex:optimizer_learning_rate':{
							'$': lr,
							'type': typeof(lr),
						},
						'ex:activation_function': activation_func
					},
					'ex:Training Data': {
						'ex:epochs': {
							'$':epochs,
							'type':typeof(epochs),
						}
					},
				}
			};
			var method = 'MulticlassClassification'
			const reply = await generateNotebook(method, objBody, content)
			console.log(reply);
			if (reply["greetings"] === 'success') {
				var path = window.location.href + '/tree/GeneratedNotebooks/MulticlassClassification.ipynb'
				console.log(path)
				const success_message = document.createElement('text');
				content.node.appendChild(success_message);
				success_message.textContent =
					'Your code has been generated successfully. Press the button below to open it.';

				const notebook_open = document.createElement('div');
				content.node.appendChild(notebook_open);
				notebook_open.innerHTML = `
								<div class="flex-container2">
									<div><button id="inputButton" type="button" onclick="window.open('`+path+`', 'MLProvCodeGen')"> Open Notebook </button></div>
								</div>
								`;
			}
        }); // end of SubmitButton event listener
		console.log(window.location.href)
    break;
	} // end switch
	});// end on the problemSelectionButton event listener
// ------------------------------------------------------------------------------------------------------------------------------- //
	// Add an application command
	const command = 'codegenerator:open';
	app.commands.addCommand(command, {
		label: 'MLProvCodeGen',
		execute: () => {
		if (!widget.isAttached) {
			// Attach content to the main work area if it's not there
			app.shell.add(widget, 'main');
		}
		// Activate the widget
		app.shell.activateById(widget.id);
		}
	});
	// Add the command to the palette.
	palette.addItem({ command, category: 'MLProvCodeGen' });	
	launcher.add({ command, category: 'Other', rank: 0 });
}
// ------------------------------------------------------------------------------------------------------------------------------- //  
// Main
const plugin: JupyterFrontEndPlugin<void> = {
  id: 'extension:plugin',
  autoStart: true,
  requires: [ICommandPalette, ILauncher],
  optional: [ISettingRegistry],
  activate: activate
}
    

export default plugin;