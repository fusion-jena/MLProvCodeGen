import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { ICommandPalette, MainAreaWidget } from '@jupyterlab/apputils';
import { Widget } from '@lumino/widgets';

import { requestAPI } from './handler';

// ------------------------------------------------------------------------------------------------------------------------------- //
//Activate
async function activate(app: JupyterFrontEnd, palette: ICommandPalette)  {
  console.log('JupyterLab extension MLProvCodeGen is activated!');
  
// ------------------------------------------------------------------------------------------------------------------------------- //
// GET request
try {
  const data = await requestAPI<any>('get_example');
  console.log(data);
} catch (reason) {
  console.error(`Error on GET /MLProvCodeGen/get_example\n${reason}`); 
} 

// ------------------------------------------------------------------------------------------------------------------------------- //
// POST request
var dataToSend = { name: 'MLProvCodeGen' };
try {
  const reply = await requestAPI<any>('get_example', {
    body: JSON.stringify(dataToSend),
    method: 'POST'
  });
  console.log(reply);
} catch (reason) {
  console.error(
    `Error on POST /MLProvCodeGen/hello ${dataToSend}.\n${reason}`  
  );
}
// ------------------------------------------------------------------------------------------------------------------------------- //
// Create a blank content widget inside of a MainAreaWidget
  const content = new Widget();
  const widget = new MainAreaWidget({ content });
  widget.id = 'MLProvCodeGen-jupyterlab';
  widget.title.label = 'MLProvCodeGen';
  widget.title.closable = true;  
// ------------------------------------------------------------------------------------------------------------------------------- //
	// Button to reset widget
	let reset = document.createElement("div");
	content.node.appendChild(reset);
	reset.innerHTML = `
		<button id="reset" type="Button"> Reset this tab </button>  
		` 
	
	reset.addEventListener('click', (event) => {
		var nodeList = content.node.childNodes; 
		console.log(nodeList);
		while (nodeList.length > 3) {
			nodeList[3].remove();
		}
	});
// ------------------------------------------------------------------------------------------------------------------------------- //
  let dropdown1 = document.createElement("div");
  content.node.appendChild(dropdown1);
  dropdown1.innerHTML = `
	<form id="dropdown1ID" onsubmit="return false">
		<label for="exercise">Choose a problem to solve:</label>
		<select name="exercise" id="exercise">
			<option value="MulticlassClassification"> Multiclass Classification using a Neural Network </option>
			<option value="ImageClassification"> Image Classification </option>
		</select>
	</form>	
  `
  // <option value="ModelSelection"> Model Selection </option>
  // <option value="Clustering"> Clustering </option>
// ------------------------------------------------------------------------------------------------------------------------------- //
  // submit button for problem selection 
    let problemButton = document.createElement("div");
	content.node.appendChild(problemButton);
	problemButton.innerHTML = `
		<button id="inputButton" type="button"> Submit </button> 
		`
// ------------------------------------------------------------------------------------------------------------------------------- //  
	// After selecting the problem, the other input units pop up
	// event listener on that input is needed for that 
	problemButton.addEventListener('click', (event)  => {
// ------------------------------------------------------------------------------------------------------------------------------- //
		// if clause for each framework (frameworks require different inputs 
		var problemSubmit = (<HTMLSelectElement>document.getElementById("exercise")).value; 
		if (problemSubmit == "ImageClassification")	{	
// ------------------------------------------------------------------------------------------------------------------------------- //
			let dropdown2 = document.createElement("div");    
			content.node.appendChild(dropdown2);
			dropdown2.innerHTML = `
				<form action="/action_page.php">
					<label for="framework">Which framework do you want to use?</label>
					<select name="framework" id="framework">
						<option value="PyTorch"> PyTorch </option>
						<option value="scikit-learn"> scikit-learn </option>
					</select>
				</form>	
				`
// ------------------------------------------------------------------------------------------------------------------------------- //
			// submit button for framework selection 
			let frameworkButton = document.createElement("div");
			content.node.appendChild(frameworkButton);
			frameworkButton.innerHTML = `
				<button id="inputButton" type="button"> Submit the framework </button> 
				`
// ------------------------------------------------------------------------------------------------------------------------------- //  
			// After selecting the framework, the other input units pop up
			// event listener on that input is needed for that 
			frameworkButton.addEventListener('click', (event)  => {
// ------------------------------------------------------------------------------------------------------------------------------- //
				// if clause for each framework (frameworks require different inputs 
				var frameworkSubmit = (<HTMLSelectElement>document.getElementById("framework")).value; 
				if (frameworkSubmit == "PyTorch")	{
// ------------------------------------------------------------------------------------------------------------------------------- //
					let dropdown3 = document.createElement("div");
					content.node.appendChild(dropdown3);
					dropdown3.innerHTML = `
						<form action="/action_page.php">
							<label for="model">Which model do you want to use?     </label>
							<select name="model" id="model">
								<option value="AlexNet"> AlexNet </option>
								<option value="ResNet"> ResNet </option>
								<option value="DenseNet"> DenseNet </option>
								<option value="VGG"> VGG </option>
							</select>
						</form>	
						` 
// ------------------------------------------------------------------------------------------------------------------------------- //
					let numberInput1 = document.createElement("div");
					content.node.appendChild(numberInput1);
					numberInput1.innerHTML = `
						<form action="/action_page.php">
							<label for="quantity">How many classes/output units?</label>
							<input type="number" id="quantity" name="quantity" value="1000"> 
							Default: 1000 classes for training on ImageNet
						</form>
						`  
// ------------------------------------------------------------------------------------------------------------------------------- //
					let checkbox1 = document.createElement("div");
					content.node.appendChild(checkbox1);
					checkbox1.innerHTML = `
						<form>
							<input type="checkbox" id="preTrainedModel" name="preTrainedModel" value="preTrainedModel">
							<label for="preTrainedModel"> Do you want to use a pre trained model?</label><br>
						</form>
						`
// ------------------------------------------------------------------------------------------------------------------------------- // 
					let dropdown4 = document.createElement("div"); 
					content.node.appendChild(dropdown4);
					dropdown4.innerHTML = `
						<form action="/action_page.php">
							<label for="data">Which data do you want to use?</label>
							<select name="data" id="data">
								<option value="Public dataset"> Public dataset </option>
								<option value="Numpy arrays"> Numpy arrays </option>
								<option value="Image files"> Image files </option>
							</select>
						</form>	
						`  
// ------------------------------------------------------------------------------------------------------------------------------- //
					let dropdown5 = document.createElement("div");
					content.node.appendChild(dropdown5);
					dropdown5.innerHTML = `
						<form action="/action_page.php">
						<label for="dataSelection">Which one?:</label>
						<select name="dataSelection" id="dataSelection">
							<option value="MNIST"> MNIST </option>
							<option value="FashionMNIST"> FashionMNIST </option>
							<option value="CIFAR10"> CIFAR10 </option>
						</select>
						</form>	
						`
// ------------------------------------------------------------------------------------------------------------------------------- //
// add pre processing text
// ------------------------------------------------------------------------------------------------------------------------------- //
					let checkbox2 = document.createElement("div");
					content.node.appendChild(checkbox2);
					checkbox2.innerHTML = `
						<form>
							<input type="checkbox" id="useGPU" name="useGPU" value="useGPU" checked>
							<label for="useGPU"> Use GPU if available? </label><br>
						</form>
						`
// ------------------------------------------------------------------------------------------------------------------------------- //
					let checkbox3 = document.createElement("div");
					content.node.appendChild(checkbox3);
					checkbox3.innerHTML = `
						<form>
							<input type="checkbox" id="modelCheckpoint" name="modelCheckpoint" value="modelCheckpoint">
							<label for="modelCheckpoint"> Save model checkpoint each epoch?</label><br>
						</form>
						`
// ------------------------------------------------------------------------------------------------------------------------------- //
					let dropdown6 = document.createElement("div");
					content.node.appendChild(dropdown6);
					dropdown6.innerHTML = `
						<form action="/action_page.php">
							<label for="lossFunc"> Loss function</label>
							<select name="lossFunc" id="lossFunc">
								<option value="CrossEntropyLoss"> CrossEntropyLoss </option>
								<option value="BCEWithLogitsLoss"> BCEWithLogitsLoss </option>
							</select>
						</form>	
						`
// ------------------------------------------------------------------------------------------------------------------------------- //
					let dropdown7 = document.createElement("div");
					content.node.appendChild(dropdown7);
					dropdown7.innerHTML = `
						<form action="/action_page.php">
							<label for="optimizer"> Optimizer </label>
							<select name="optimizer" id="optimizer">
								<option value="Adam"> Adam </option>
								<option value="Adadelta"> Adadelta </option>
								<option value="Adagrad"> Adagrad </option>
								<option value="Adamax"> Adamax </option>
								<option value="RMSprop"> RMSprop </option>
								<option value="SGD"> SGD </option>
							</select>
						</form>	
						`
// ------------------------------------------------------------------------------------------------------------------------------- //
					let numberInput2 = document.createElement("div");
					content.node.appendChild(numberInput2);
					numberInput2.innerHTML = `
						<form action="/action_page.php">
							<label for="rate"> Learning rate</label>
							<input type="number" id="rate" name="rate" value="0.001">
						</form>
						`
// ------------------------------------------------------------------------------------------------------------------------------- //
					let numberInput3 = document.createElement("div");
					content.node.appendChild(numberInput3);
					numberInput3.innerHTML = `
						<form action="/action_page.php">
							<label for="batches"> Batch Size</label>
							<input type="number" id="batches" name="batches" value="128">
						</form>
						`
// ------------------------------------------------------------------------------------------------------------------------------- //
					let numberInput4 = document.createElement("div");
					content.node.appendChild(numberInput4);
					numberInput4.innerHTML = `
						<form action="/action_page.php">
							<label for="epochs">How many epochs?</label>
							<input type="number" id="epochs" name="epochs" value="3">
						</form>
						`
// ------------------------------------------------------------------------------------------------------------------------------- //
					let numberInput5 = document.createElement("div");
					content.node.appendChild(numberInput5);
					numberInput5.innerHTML = `
						<form action="/action_page.php">
							<label for="printProgress"> Print progress every ... batches</label>
							<input type="number" id="printProgress" name="printProgress" value="1">
						</form>
						`
// ------------------------------------------------------------------------------------------------------------------------------- //
					let dropdown8 = document.createElement("div");
					content.node.appendChild(dropdown8);
					dropdown8.innerHTML = `
						<form action="/action_page.php">
							<label for="logs"> How to log metrics </label>
							<select name="logs" id="logs">
								<option value="notAtAll"> Not at all </option>
								<option value="Tensorboard"> Tensorboard </option>
								<option value="Aim"> Aim </option>
								<option value="Weights & Biases"> weightsAndBiases </option>
								<option value="comet.ml"> comet.ml </option>
							</select>
						</form>	
						`
// ------------------------------------------------------------------------------------------------------------------------------- //  
					let submitButton = document.createElement("div");
					content.node.appendChild(submitButton);
					submitButton.innerHTML = `
						<button id="inputButton" type="button"> Submit your values </button>  
						`
// ------------------------------------------------------------------------------------------------------------------------------- //  	
					submitButton.addEventListener('click', async (event)  => {
						// dropdowns 
						var exerciseValue = (<HTMLSelectElement>document.getElementById("exercise")).value;
						var frameworkValue = (<HTMLSelectElement>document.getElementById("framework")).value;
						var modelValue = (<HTMLSelectElement>document.getElementById("model")).value;
						var dataValue = (<HTMLSelectElement>document.getElementById("data")).value;
						var dataSelectionValue = (<HTMLSelectElement>document.getElementById("dataSelection")).value;
						var lossFuncValue = (<HTMLSelectElement>document.getElementById("lossFunc")).value;
						var optimizerValue = (<HTMLSelectElement>document.getElementById("optimizer")).value;
						var logsValue = (<HTMLSelectElement>document.getElementById("logs")).value;
						// numbers
						var quantityValue = (<HTMLInputElement>document.getElementById("quantity")).value;
						var rateValue = (<HTMLInputElement>document.getElementById("rate")).value;
						var batchesValue = (<HTMLInputElement>document.getElementById("batches")).value;
						var epochsValue = (<HTMLInputElement>document.getElementById("epochs")).value;
						var printProgressValue = (<HTMLInputElement>document.getElementById("printProgress")).value;
						// checkboxes
						var preTrainedModelValue = 3;
						var useGPUValue = 3;
						var modelCheckpointValue = 3;
						if ((<HTMLInputElement>document.getElementById("preTrainedModel")).checked) {
							preTrainedModelValue = 1;
						} else {	preTrainedModelValue = 0;	}
		
						if ((<HTMLInputElement>document.getElementById("useGPU")).checked) {
									useGPUValue = 1;
						} else {	useGPUValue = 0;	}
				
						if ((<HTMLInputElement>document.getElementById("modelCheckpoint")).checked) {
									modelCheckpointValue = 1;
						} else {	modelCheckpointValue = 0;	} 	
// ------------------------------------------------------------------------------------------------------------------------------- //  
						// convert variables into JSON/ input Object     
						const objBody = {
							// dropdowns
							"exercise": exerciseValue, 
							"framework": frameworkValue,
							"model": modelValue,
							"data": dataValue,
							"data_selection": dataSelectionValue,
							"loss_function": lossFuncValue,
							"optimizer": optimizerValue,
							"visualization_tool": logsValue,
							// numbers
							"quantity": quantityValue,
							"rate": rateValue,
							"batches": batchesValue,
							"epochs": epochsValue,
							"print_progress": printProgressValue,
							// checkboxes
							"pre_trained_model": preTrainedModelValue,
							"use_GPU": useGPUValue,
							"model_checkpoint": modelCheckpointValue
						}      
// ------------------------------------------------------------------------------------------------------------------------------- //  		
						// Post request with input data 
						try {
								const reply = await requestAPI<any>('ImageClassification_pytorch', {
									body: JSON.stringify(objBody),
									method: 'POST'
								});
								console.log(reply);
// ------------------------------------------------------------------------------------------------------------------------------- // 
								if (reply["greetings"] == "success") {
									let success_message = document.createElement("text");
									content.node.appendChild(success_message);
									success_message.textContent = "Your Code has been generated successfully. Press the button below to open it."
					
									let notebook_open = document.createElement("div");
									content.node.appendChild(notebook_open);
									notebook_open.innerHTML = `
										<button id="inputButton" type="button" onclick="window.open('http://localhost:8888/notebooks/MLProvCodeGen/ImageClassification_PyTorch.ipynb')"> Open Notebook </button>  
										`
									}
// ------------------------------------------------------------------------------------------------------------------------------- // 
						} catch (reason) {
							console.error(
								`Error on POST /MLProvCodeGen/ImageClassification_pytorch ${dataToSend}.\n${reason}`
							);
						}
					});
// ------------------------------------------------------------------------------------------------------------------------------- // 
				// End on the frameworkButton event listener  AND end of if
				}
// ------------------------------------------------------------------------------------------------------------------------------- // 
		// else for image classification --> scikit-learn
				else if (frameworkSubmit == "scikit-learn")	{
// ------------------------------------------------------------------------------------------------------------------------------- //
					let scikit_model = document.createElement("div");
					content.node.appendChild(scikit_model);
					scikit_model.innerHTML = `
						<form action="/action_page.php">
							<label for="model_func"> Which model </label>
							<select name="model_func" id="model_func">
								<option value="sklearn.svm.SVC"> Support vectors </option>
								<option value="sklearn.ensemble.RandomForestClassifier"> Random forest </option>
								<option value="sklearn.linear_model.Perceptron"> Perceptron </option>
								<option value="sklearn.neighbors.KNeighborsClassifier"> K-nearest neighbors </option>
								<option value="sklearn.tree.DecisionTreeClassifier"> Decision tree </option>
							</select>
						</form>	
						`
// ------------------------------------------------------------------------------------------------------------------------------- //
					let scikit_input_data = document.createElement("div");
					content.node.appendChild(scikit_input_data);
					scikit_input_data.innerHTML = `
						<form action="/action_page.php">
							<label for="data_format"> What best describes your input data? </label>
							<select name="data_format" id="data_format">
								<option value="Numpy arrays"> Numpy arrays </option>
								<option value="Image files"> Image files </option>
							</select>
						</form>	
						`
// ------------------------------------------------------------------------------------------------------------------------------- //
					let checkbox1 = document.createElement("div");
					content.node.appendChild(checkbox1);
					checkbox1.innerHTML = `
						<form>
							<input type="checkbox" id="scale_mean_std" name="scale_mean_std" value="scale_mean_std" checked>
							<label for="scale_mean_std"> Scale to mean 0, std1 </label><br>
						</form>
						`
// ------------------------------------------------------------------------------------------------------------------------------- //
					let scikit_visualization = document.createElement("div");
					content.node.appendChild(scikit_visualization);
					scikit_visualization.innerHTML = `
						<form action="/action_page.php">
							<label for="visualization_tool"> How to log metrics? </label>
							<select name="visualization_tool" id="visualization_tool">
								<option value="0"> Not at all </option>
								<option value="Tensorboard"> Tensorboard </option>
								<option value="comet.ml"> comet.ml </option>
							</select>
						</form>	
						`	
// ------------------------------------------------------------------------------------------------------------------------------- //
					let cometAPIKey = document.createElement("div");
					content.node.appendChild(cometAPIKey);
					cometAPIKey.innerHTML = `
						<form action="/action_page.php">
							<label for="APIKey"> Enter your Comet API Key here (required, if you want to use Comet.ml for logging) </label>
							<input type="text" id="APIKey" name="APIKey">
						</form>	
						`
// ------------------------------------------------------------------------------------------------------------------------------- //
					let cometName = document.createElement("div");
					content.node.appendChild(cometName);
					cometName.innerHTML = `
						<form action="/action_page.php">
							<label for="cometName"> Enter your Comet project name here (optional) </label>
							<input type="text" id="cometName" name="cometName">
						</form>	
						`					
// ------------------------------------------------------------------------------------------------------------------------------- //  
					let submitButton = document.createElement("div");
					content.node.appendChild(submitButton);
					submitButton.innerHTML = `
						<button id="inputButton" type="button"> Submit your values </button>  
						`
// ------------------------------------------------------------------------------------------------------------------------------- //
					submitButton.addEventListener('click', async (event)  => {
						// var setup  
						var model_func = (<HTMLSelectElement>document.getElementById("model_func")).value;
						var data_format = (<HTMLSelectElement>document.getElementById("data_format")).value;
						var scale_mean_std = (<HTMLSelectElement>document.getElementById("scale_mean_std")).value;
						var visualization_tool = (<HTMLSelectElement>document.getElementById("visualization_tool")).value;
						var cometAPIKey = (<HTMLInputElement>document.getElementById("APIKey")).value;
						var cometName = (<HTMLInputElement>document.getElementById("cometName")).value;
// ------------------------------------------------------------------------------------------------------------------------------- //  
						// convert variables into JSON/ input Object     
						const objBody = {
							// dropdowns
							"model_func": model_func, 
							"data_format": data_format,
							"scale_mean_std": scale_mean_std,
							"visualization_tool": visualization_tool,
							"cometAPIKey": cometAPIKey,
							"cometName": cometName
						}
// ------------------------------------------------------------------------------------------------------------------------------- //  		
						// Post request with input data 
						try {
							const reply = await requestAPI<any>('ImageClassification_scikit', { 
								body: JSON.stringify(objBody),
								method: 'POST'
							});
							console.log(reply);
							if (reply["greetings"] == "success") {
									let success_message = document.createElement("text");
									content.node.appendChild(success_message);
									success_message.textContent = "Your Code has been generated successfully. Press the button below to open it."
					
									let notebook_open = document.createElement("div");
									content.node.appendChild(notebook_open);
									notebook_open.innerHTML = `
										<button id="inputButton" type="button" onclick="window.open('http://localhost:8888/notebooks/MLProvCodeGen/ImageClassification_Scikit.ipynb')"> Open Notebook </button>  
										`
									}
						}	catch (reason) {
							console.error(
								`Error on POST /MLProvCodeGen/ImageClassification_scikit ${dataToSend}.\n${reason}`
							);
						}
					});	
// ------------------------------------------------------------------------------------------------------------------------------- //
				// endif		
				}
// ------------------------------------------------------------------------------------------------------------------------------- // 
			// end on the frameworkButton event listener   
			});
// ------------------------------------------------------------------------------------------------------------------------------- //
		// endif problemSubmit, start clustering
		}	else	if (problemSubmit == "Clustering")	{
// ------------------------------------------------------------------------------------------------------------------------------- //
				let clustering_framework = document.createElement("div");    
				content.node.appendChild(clustering_framework);
				clustering_framework.innerHTML = `
					<form action="/action_page.php">
						<label for="framework">Which framework do you want to use?</label>
						<select name="framework" id="framework">
							<option value="scikit-learn"> scikit-learn </option>
							<option value="PyTorch"> PyTorch </option>
						</select>
					</form>	
					`
// ------------------------------------------------------------------------------------------------------------------------------- //
			// submit button for framework selection 
			let frameworkButton = document.createElement("div");
			content.node.appendChild(frameworkButton);
			frameworkButton.innerHTML = `
				<button id="inputButton" type="button"> Submit the framework </button> 
				`
// ------------------------------------------------------------------------------------------------------------------------------- //  
			// After selecting the framework, the other input units pop up
			// event listener on that input is needed for that 
			frameworkButton.addEventListener('click', (event)  => {
// ------------------------------------------------------------------------------------------------------------------------------- //
				// if clause for each framework (frameworks require different inputs 
				var frameworkSubmit = (<HTMLSelectElement>document.getElementById("framework")).value; 
				if (frameworkSubmit == "scikit-learn")	{
// ------------------------------------------------------------------------------------------------------------------------------- //
					let scikit_model = document.createElement("div");
					content.node.appendChild(scikit_model);
					scikit_model.innerHTML = `
						<form action="/action_page.php">
							<label for="model_func"> Which model </label>
							<select name="model_func" id="model_func">
								<option value="K-means"> K-means </option>
								<option value="DBSCAN"> DBSCAN </option>
								<option value="Gauss"> Gaussian Mixture </option>
							</select>
						</form>	
						`
// ------------------------------------------------------------------------------------------------------------------------------- //
					let scikit_data_format = document.createElement("div");
					content.node.appendChild(scikit_data_format);
					scikit_data_format.innerHTML = `
						<form action="/action_page.php">
							<label for="data_format"> Dataset </label>
							<select name="data_format" id="data_format">
								<option value="example"> Iris example </option>
								<option value="Spiral"> Spiral </option>
								<option value="Aggregation"> Aggregation </option>
								<option value="R15"> R15 </option>
								<option value="own"> Use your own data </option>
							</select>
							<p><i> If you want to use your own data then select the corresponding option here and follow the instructions in the generated notebook. <\i><\p>
						</form>	
						`
// ------------------------------------------------------------------------------------------------------------------------------- //
					let preprocessing_check = document.createElement("div");
					content.node.appendChild(preprocessing_check);
					preprocessing_check.innerHTML = `
						<form>
							<input type="checkbox" id="preprocessing" name="preprocessing" value="1" checked>
							<label for="preprocessing"> Use a Min-Max-Scaler? (Recommended) </label><br>
						</form>
						`			
// ------------------------------------------------------------------------------------------------------------------------------- //  
					let submitButton = document.createElement("div");
					content.node.appendChild(submitButton);
					submitButton.innerHTML = `
						<button id="inputButton" type="button"> Submit your values </button>  
						` 
// ------------------------------------------------------------------------------------------------------------------------------- //
					submitButton.addEventListener('click', async (event)  => {
// ------------------------------------------------------------------------------------------------------------------------------- //
						// var setup  
						var model_func = (<HTMLSelectElement>document.getElementById("model_func")).value;
						var data_format = (<HTMLSelectElement>document.getElementById("data_format")).value;
						var preprocessing = (<HTMLInputElement>document.getElementById("preprocessing")).checked;
// ------------------------------------------------------------------------------------------------------------------------------- //  
						// convert variables into JSON/ input Object     
						const objBody = {
							// dropdowns
							"model_func": model_func, 
							"data_format": data_format,
							"preprocessing": preprocessing
						}
// ------------------------------------------------------------------------------------------------------------------------------- //  		
						// Post request with input data 
						try {
							const reply = await requestAPI<any>('Clustering_scikit', { 
								body: JSON.stringify(objBody),
								method: 'POST'
							});
							console.log(reply);
							console.log(reply["greetings"]);
							if (reply["greetings"] == "success") {
								let success_message = document.createElement("text");
								content.node.appendChild(success_message);
								success_message.textContent = "Your Code has been generated successfully. Press the button below to open it."
					
								let notebook_open = document.createElement("div");
								content.node.appendChild(notebook_open);
								notebook_open.innerHTML = `
									<button id="inputButton" type="button" onclick="window.open('http://localhost:8888/notebooks/MLProvCodeGen/userInputNotebook.ipynb')"> Open Notebook </button>  
									`
					
							}
						}	catch (reason) {
							console.error(
								`Error on POST /MLProvCodeGen/Clustering_scikit ${dataToSend}.\n${reason}`
							);
						}
					});	
// ------------------------------------------------------------------------------------------------------------------------------- //			
				}	else	if (frameworkSubmit == "PyTorch") {
				}
// ------------------------------------------------------------------------------------------------------------------------------- // 
			// end on the frameworkButton event listener   
			});  
// ------------------------------------------------------------------------------------------------------------------------------- // 
		} else	if (problemSubmit == "ModelSelection")	{
// ------------------------------------------------------------------------------------------------------------------------------- //
			let MS_framework = document.createElement("div");    
			content.node.appendChild(MS_framework);
			MS_framework.innerHTML = `
				<form action="/action_page.php">
					<label for="framework">Which framework do you want to use?</label>
					<select name="framework" id="framework">
						<option value="scikit-learn"> scikit-learn (currently the only supported framework) </option>
					</select>
				</form>	
				`
// ------------------------------------------------------------------------------------------------------------------------------- //
			let MS_model = document.createElement("div");
			content.node.appendChild(MS_model);
			MS_model.innerHTML = `
				<form action="/action_page.php">
					<label for="model">Which model do you want to use?     </label>
					<select name="model" id="model">
						<option value="alexnet"> AlexNet </option>
						<option value="resnet"> ResNet </option>
						<option value="densenet"> DenseNet </option>
					</select>
				</form>	
				` 
// ------------------------------------------------------------------------------------------------------------------------------- //
			let classes = document.createElement("div");
			content.node.appendChild(classes);
			classes.innerHTML = `
				<form action="/action_page.php">
					<label for="quantity">How many classes/output units?</label>
					<input type="number" id="quantity" name="quantity" value="1000"> 
					Default: 1000 classes for training on ImageNet
				</form>
				`  
// ------------------------------------------------------------------------------------------------------------------------------- //
			let preTrainedModel = document.createElement("div");
			content.node.appendChild(preTrainedModel);
			preTrainedModel.innerHTML = `
				<form>
					<input type="checkbox" id="preTrainedModel" name="preTrainedModel" value="preTrainedModel">
					<label for="preTrainedModel"> Do you want to use a pre trained model?</label><br>
				</form>
				`
// ------------------------------------------------------------------------------------------------------------------------------- // 
			let MS_data_format = document.createElement("div"); 
			content.node.appendChild(MS_data_format);
			MS_data_format.innerHTML = `
				<form action="/action_page.php">
					<label for="data">Which data do you want to use?</label>
					<select name="data" id="data">
						<option value="Public dataset"> Public dataset </option>
						<option value="Numpy arrays"> Numpy arrays </option>
						<option value="Image files"> Image files </option>
					</select>
				</form>	
				`  
// ------------------------------------------------------------------------------------------------------------------------------- //
			let MS_dataset = document.createElement("div");
			content.node.appendChild(MS_dataset);
			MS_dataset.innerHTML = `
				<form action="/action_page.php">
				<label for="dataSelection">Which one?:</label>
				<select name="dataSelection" id="dataSelection">
					<option value="MNIST"> MNIST </option>
					<option value="FashionMNIST"> FashionMNIST </option>
				</select>
				</form>	
				`
// ------------------------------------------------------------------------------------------------------------------------------- //
// add pre processing text
// ------------------------------------------------------------------------------------------------------------------------------- //
			let GPU = document.createElement("div");
			content.node.appendChild(GPU);
			GPU.innerHTML = `
				<form>
					<input type="checkbox" id="useGPU" name="useGPU" value="useGPU" checked>
					<label for="useGPU"> Use GPU if available? </label><br>
				</form>
				`
// ------------------------------------------------------------------------------------------------------------------------------- //
			let modelCheckpoint = document.createElement("div");
			content.node.appendChild(modelCheckpoint);
			modelCheckpoint.innerHTML = `
				<form>
					<input type="checkbox" id="modelCheckpoint" name="modelCheckpoint" value="modelCheckpoint">
					<label for="modelCheckpoint"> Save model checkpoint each epoch?</label><br>
				</form>
				`
// ------------------------------------------------------------------------------------------------------------------------------- //
			let MS_loss = document.createElement("div");
			content.node.appendChild(MS_loss);
			MS_loss.innerHTML = `
				<form action="/action_page.php">
					<label for="lossFunc"> Loss function</label>
					<select name="lossFunc" id="lossFunc">
						<option value="CrossEntropyLoss"> CrossEntropyLoss </option>
						<option value="BCEWithLogitsLoss"> BCEWithLogitsLoss </option>
					</select>
				</form>	
				`
// ------------------------------------------------------------------------------------------------------------------------------- //
			let MS_optimizer = document.createElement("div");
			content.node.appendChild(MS_optimizer);
			MS_optimizer.innerHTML = `
				<form action="/action_page.php">
					<label for="optimizer"> Optimizer </label>
					<select name="optimizer" id="optimizer">
						<option value="Adam"> Adam </option>
						<option value="Adadelta"> Adadelta </option>
						<option value="Adagrad"> Adagrad </option>
						<option value="Adamax"> Adamax </option>
						<option value="RMSprop"> RMSprop </option>
						<option value="SGD"> SGD </option>
					</select>
				</form>	
				`
// ------------------------------------------------------------------------------------------------------------------------------- //
			let lr = document.createElement("div");
			content.node.appendChild(lr);
			lr.innerHTML = `
				<form action="/action_page.php">
					<label for="rate"> Learning rate</label>
					<input type="number" id="rate" name="rate" value="0.001">
				</form>
				`
// ------------------------------------------------------------------------------------------------------------------------------- //
			let batches = document.createElement("div");
			content.node.appendChild(batches);
			batches.innerHTML = `
				<form action="/action_page.php">
					<label for="batches"> Batch Size</label>
					<input type="number" id="batches" name="batches" value="128">
				</form>
				`
// ------------------------------------------------------------------------------------------------------------------------------- //
			let epochs = document.createElement("div");
			content.node.appendChild(epochs);
			epochs.innerHTML = `
				<form action="/action_page.php">
					<label for="epochs">How many epochs?</label>
					<input type="number" id="epochs" name="epochs" value="3">
				</form>
				`
// ------------------------------------------------------------------------------------------------------------------------------- //
			let printProgress = document.createElement("div");
			content.node.appendChild(printProgress);
			printProgress.innerHTML = `
				<form action="/action_page.php">
					<label for="printProgress"> Print progress every ... batches</label>
					<input type="number" id="printProgress" name="printProgress" value="1">
				</form>
				`
// ------------------------------------------------------------------------------------------------------------------------------- //
			let MS_visualization_tool = document.createElement("div");
			content.node.appendChild(MS_visualization_tool);
			MS_visualization_tool.innerHTML = `
				<form action="/action_page.php">
					<label for="logs"> How to log metrics </label>
					<select name="logs" id="logs">
						<option value="notAtAll"> Not at all </option>
						<option value="Tensorboard"> Tensorboard </option>
						<option value="Aim"> Aim </option>
						<option value="Weights & Biases"> weightsAndBiases </option>
						<option value="comet.ml"> comet.ml </option>
					</select>
				</form>	
				`
// ------------------------------------------------------------------------------------------------------------------------------- //  
			let submitButton = document.createElement("div");
			content.node.appendChild(submitButton);
			submitButton.innerHTML = `
				<button id="inputButton" type="button"> Submit your values </button>  
				`
			submitButton.addEventListener('click', async (event)  => {
				// dropdowns 
				var exerciseValue = (<HTMLSelectElement>document.getElementById("exercise")).value;
				var frameworkValue = (<HTMLSelectElement>document.getElementById("framework")).value;
				var modelValue = (<HTMLSelectElement>document.getElementById("model")).value;
				var dataValue = (<HTMLSelectElement>document.getElementById("data")).value;
				var dataSelectionValue = (<HTMLSelectElement>document.getElementById("dataSelection")).value;
				var lossFuncValue = (<HTMLSelectElement>document.getElementById("lossFunc")).value;
				var optimizerValue = (<HTMLSelectElement>document.getElementById("optimizer")).value;
				var logsValue = (<HTMLSelectElement>document.getElementById("logs")).value;
				// numbers
				var quantityValue = (<HTMLInputElement>document.getElementById("quantity")).value;
				var rateValue = (<HTMLInputElement>document.getElementById("rate")).value;
				var batchesValue = (<HTMLInputElement>document.getElementById("batches")).value;
				var epochsValue = (<HTMLInputElement>document.getElementById("epochs")).value;
				var printProgressValue = (<HTMLInputElement>document.getElementById("printProgress")).value;
				// checkboxes
				var preTrainedModelValue = 3;
				var useGPUValue = 3;
				var modelCheckpointValue = 3;
				if ((<HTMLInputElement>document.getElementById("preTrainedModel")).checked) {
					preTrainedModelValue = 1;
				} else {	preTrainedModelValue = 0;	}

				if ((<HTMLInputElement>document.getElementById("useGPU")).checked) {
							useGPUValue = 1;
				} else {	useGPUValue = 0;	}
				
				if ((<HTMLInputElement>document.getElementById("modelCheckpoint")).checked) {
							modelCheckpointValue = 1;
				} else {	modelCheckpointValue = 0;	} 	
// ------------------------------------------------------------------------------------------------------------------------------- //  
				// convert variables into JSON/ input Object     
				const objBody = {
					// dropdowns
					"exercise": exerciseValue, 
					"framework": frameworkValue,
					"model": modelValue,
					"data": dataValue,
					"data_selection": dataSelectionValue,
					"loss_function": lossFuncValue,
					"optimizer": optimizerValue,
					"visualization_tool": logsValue,
					// numbers
					"quantity": quantityValue,
					"rate": rateValue,
					"batches": batchesValue,
					"epochs": epochsValue,
					"print_progress": printProgressValue,
					// checkboxes
					"pre_trained_model": preTrainedModelValue,
					"use_GPU": useGPUValue,
					"model_checkpoint": modelCheckpointValue
				}      
// ------------------------------------------------------------------------------------------------------------------------------- //  		
				// Post request with input data 
				try {
						const reply = await requestAPI<any>('ModelSelection_scikit', {
							body: JSON.stringify(objBody),
							method: 'POST'
						});
						console.log(reply);
// ------------------------------------------------------------------------------------------------------------------------------- // 
						if (reply["greetings"] == "success") {
							let success_message = document.createElement("text");
							content.node.appendChild(success_message);
							success_message.textContent = "Your Code has been generated successfully. Press the button below to open it."
					
							let notebook_open = document.createElement("div");
							content.node.appendChild(notebook_open);
							notebook_open.innerHTML = `
								<button id="inputButton" type="button" onclick="window.open('http://localhost:8888/notebooks/MLProvCodeGen/ModelSelection_scikit.ipynb')"> Open Notebook </button>  
								`
							}
// ------------------------------------------------------------------------------------------------------------------------------- // 
					} catch (reason) {
						console.error(
							`Error on POST /MLProvCodeGen/ModelSelection_scikit ${dataToSend}.\n${reason}`
						);
					}
				});
// ------------------------------------------------------------------------------------------------------------------------------- // 
		// end of exercise selection if loop
		}	else	if (problemSubmit == "MulticlassClassification")	{
				let clustering_framework = document.createElement("div");    
				content.node.appendChild(clustering_framework);
				clustering_framework.innerHTML = `
					<form action="/action_page.php">
						<label for="framework">Which framework do you want to use?</label>
						<select name="framework" id="framework">
							<option value="PyTorch"> PyTorch (currently the only option for model training)</option>
						</select>
					</form>	
					`
// ------------------------------------------------------------------------------------------------------------------------------- //
				// submit button for framework selection 
				let frameworkButton = document.createElement("div");
				content.node.appendChild(frameworkButton);
				frameworkButton.innerHTML = `
					<button id="inputButton" type="button"> Submit the framework </button> 
					`
// ------------------------------------------------------------------------------------------------------------------------------- //  
				// After selecting the framework, the other input units pop up
				// event listener on that input is needed for that 
				frameworkButton.addEventListener('click', (event)  => {
// ------------------------------------------------------------------------------------------------------------------------------- //  					
					// UI Inputs
					let dataHeader = document.createElement("div");
					content.node.appendChild(dataHeader);
					dataHeader.innerHTML = `
						<b><u> Data Settings</u></b>
					`
					
					let dataset = document.createElement("div");
					content.node.appendChild(dataset);
					dataset.innerHTML = `
						<form action="/action_page.php">
							<label for="dataset">Which dataset do you want to use?</label>
							<select name="dataset" id="dataset">
								<option value="Iris"> Iris </option>
								<option value="Spiral"> Spiral </option>
								<option value="Aggregation"> Aggregation </option>
								<option value="R15"> R15 </option>
								<option value="User"> Use your own Data (.csv) </option>
							</select>
						</form>	
						`
					let random_seed = document.createElement("div");
					content.node.appendChild(random_seed);
					random_seed.innerHTML = `
						<form action="/action_page.php">
							<label for="random_seed">Random Seed for data Segregation (default: 2)</label>
							<input type="number" id="random_seed" name="random_seed" value="2">
						</form>
						`	
					
					let test_split = document.createElement("div");
					content.node.appendChild(test_split);
					test_split.innerHTML = `
						<form action="/action_page.php">
							<label for="test_split">Testing data split (default: 0.2)</label>
							<input type="number" id="test_split" name="test_split" value="0.2">
						</form>
						`
					
					let preprocessing_text = document.createElement("div");
					content.node.appendChild(preprocessing_text);
					preprocessing_text.innerHTML = `
						<label> <i>preprocessing: MinMaxScaler</i></label>
					`	
					
					let modelHeader = document.createElement("div");
					content.node.appendChild(modelHeader);
					modelHeader.innerHTML = `
						<b><u> Model Settings</u></b>
					`	
					
					let activation_func = document.createElement("div");
					content.node.appendChild(activation_func);
					activation_func.innerHTML = `
						<form action="/action_page.php">
							<label for="activation_func">Activation function:</label>
							<select name="activation_func" id="activation_func">
								<option value="F.softmax(self.layer3(x), dim=1)"> Softmax </option>
								<option value="torch.sigmoid(self.layer3(x))"> Sigmoid </option>
								<option value="torch.tanh(self.layer3(x))"> Tanh </option>
							</select>
						</form>	
						`
					
					let neuron_number = document.createElement("div");
					content.node.appendChild(neuron_number);
					neuron_number.innerHTML = `
						<form action="/action_page.php">
							<label for="neuron_number">How many Neurons per linear layer? (Input and output neurons are separate) </label>
							<input type="number" id="neuron_number" name="neuron_number" value="50">
						</form>
						`
					
					let epochs = document.createElement("div");
					content.node.appendChild(epochs);
					epochs.innerHTML = `
						<form action="/action_page.php">
							<label for="epochs">How many Epochs?</label>
							<input type="number" id="epochs" name="epochs" value="100">
						</form>
						`
						
					let NN_optimizer = document.createElement("div");
					content.node.appendChild(NN_optimizer);
					NN_optimizer.innerHTML = `
						<form action="/action_page.php">
							<label for="optimizer"> Optimizer </label>
							<select name="optimizer" id="optimizer">
								<option value="torch.optim.Adam("> Adam </option>
								<option value="torch.optim.Adadelta("> Adadelta </option>
								<option value="torch.optim.Adagrad("> Adagrad </option>
								<option value="torch.optim.Adamax("> Adamax </option>
								<option value="torch.optim.RMSprop("> RMSprop </option>
								<option value="torch.optim.SGD("> SGD </option>
							</select>
						</form>	
						`
					let default_lr = document.createElement("div");
					content.node.appendChild(default_lr);
					default_lr.innerHTML = `
						<form>
							<input type="checkbox" id="default" name="default" value="default" checked>
							<label for="default"> Use optimizers default learning rate? </label><br>
						</form>
						`
					
					let lr = document.createElement("div");
					content.node.appendChild(lr);
					lr.innerHTML = `
						<form action="/action_page.php">
							<label for="rate"> Learning rate</label>
							<input type="number" id="rate" name="rate" value="0.001">
						</form>
						`
					let MS_loss = document.createElement("div");
					content.node.appendChild(MS_loss);
					MS_loss.innerHTML = `
						<form action="/action_page.php">
							<label for="lossFunc"> Loss function</label>
							<select name="lossFunc" id="lossFunc">
								<option value="nn.CrossEntropyLoss()"> CrossEntropyLoss </option>
								<option value="nn.NLLLoss()"> NLLLoss </option>
								<option value="nn.MultiMarginLoss()"> MultiMarginLoss </option>
							</select>
						</form>	
						`
					let use_gpu = document.createElement("div");
					content.node.appendChild(use_gpu);
					use_gpu.innerHTML = `
						<form>
							<input type="checkbox" id="use_gpu" name="use_gpu" value="use_gpu" checked>
							<label for="use_gpu"> Use GPU if available? </label><br>
						</form>
						`
						
					let submitButton = document.createElement("div");
					content.node.appendChild(submitButton);
					submitButton.innerHTML = `
						<button id="inputButton" type="button"> Submit your values </button>  
						`
					submitButton.addEventListener('click', async (event)  => {
// ------------------------------------------------------------------------------------------------------------------------------- //  
						// Get Variables
						var exercise = (<HTMLSelectElement>document.getElementById("exercise")).value;
						var framework = (<HTMLSelectElement>document.getElementById("framework")).value;
						var dataset = (<HTMLSelectElement>document.getElementById("dataset")).value;
						var activation_func = (<HTMLSelectElement>document.getElementById("activation_func")).value;
						var optimizer = (<HTMLSelectElement>document.getElementById("optimizer")).value;
						var loss_func = (<HTMLSelectElement>document.getElementById("lossFunc")).value;
						
						var test_split = (<HTMLInputElement>document.getElementById("test_split")).value;
						var neuron_number = (<HTMLInputElement>document.getElementById("neuron_number")).value;
						var epochs = (<HTMLInputElement>document.getElementById("epochs")).value;
						var lr = (<HTMLInputElement>document.getElementById("rate")).value;
						var random_seed = (<HTMLInputElement>document.getElementById("random_seed")).value;
						
						var defaultValue, use_gpu;
						if ((<HTMLInputElement>document.getElementById("default")).checked) {
							defaultValue = 1;
						} else {	defaultValue = 0;	} 	
						
						if ((<HTMLInputElement>document.getElementById("use_gpu")).checked) {
									use_gpu = 1;
						} else {	use_gpu = 0;	}
// ------------------------------------------------------------------------------------------------------------------------------- //  
						// convert variables into JSON/ input Object     
						const objBody = {
							"exercise": exercise,
							"framework": framework,
							"dataset": dataset,
							"random_seed": random_seed,
							"test_split": test_split,
							"activation_func": activation_func,
							"neuron_number": neuron_number,
							"optimizer": optimizer,
							"loss_func": loss_func,
							"epochs": epochs,
							"lr": lr,
							"use_gpu": use_gpu,
							"default": defaultValue
						}
						// Post request with input data 
						try {
								const reply = await requestAPI<any>('MulticlassClassification', {
									body: JSON.stringify(objBody),
									method: 'POST'
								});
								console.log(reply);
// ------------------------------------------------------------------------------------------------------------------------------- // 
								if (reply["greetings"] == "success") {
									let success_message = document.createElement("text");
									content.node.appendChild(success_message);
									success_message.textContent = "Your Code has been generated successfully. Press the button below to open it."
							
									let notebook_open = document.createElement("div");
									content.node.appendChild(notebook_open);
									notebook_open.innerHTML = `
										<button id="inputButton" type="button" onclick="window.open('http://localhost:8888/notebooks/MLProvCodeGen/MulticlassClassification.ipynb')"> Open Notebook </button>  
										`
									}
// ------------------------------------------------------------------------------------------------------------------------------- // 
							} catch (reason) {
								console.error(
									`Error on POST /MLProvCodeGen/MulticlassClassification ${dataToSend}.\n${reason}`
								);
							}
						});
				});
		}
// ------------------------------------------------------------------------------------------------------------------------------- // 
			// end on the problemButton event listener   
	});  	
// ------------------------------------------------------------------------------------------------------------------------------- //  
	// Add an application command
	const command: string = 'codegenerator:open';
	app.commands.addCommand(command, {
		label: 'Code Generation from Provenance data',
		execute: () => {
      
		if (!widget.isAttached) {
			// Attach content to the main work area if it's not there 
			app.shell.add(widget, 'main');
		} 
		// Activate the widget
		app.shell.activateById(widget.id);
		}
	});
// ------------------------------------------------------------------------------------------------------------------------------- //
	// Add the command to the palette.
		palette.addItem({ command, category: 'Tutorial' });
}
	
/**
 * Initialization data for the codegenerator extension.
 */
const extension: JupyterFrontEndPlugin<void> = {
  id: 'MLProvCodeGen',
  autoStart: true,
  requires: [ICommandPalette],
  activate: activate
};

export default extension;