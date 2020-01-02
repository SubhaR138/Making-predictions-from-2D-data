//load and visualize the data
async function getData(){

    //fetching the cars dataset

    const carsDataReq=await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');

    /*load rthe cars dataset from json file .This data contains many features
    *about cars,we only extract mpg and horsepower.*/

    const carsData=await carsDataReq.json();

    /*map() call the provided function once for each element in an array*/

    const cleaned = carsData.map(car => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower,
      }))

      //filter()creates an array filled with all array elements that pass a test.

      .filter(car => (car.mpg != null && car.horsepower != null));
      return cleaned;
    }
    //the above coding wil remove any entries that do not have mpg or horsepower defined.

    //plotting the data in scatterplot to see how it looks like
    async function run() {
        // Load and plot the original input data that we are going to train on.
        const data = await getData();
        const values = data.map(d => ({
          x: d.horsepower,
          y: d.mpg,
        }));
      
        tfvis.render.scatterplot(
          {name: 'Horsepower v MPG'},
          {values}, 
          {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300
          }
        );

        // Create the model
        //this will show a summary of the layers on webpage.
         const model = createModel();  
         tfvis.show.modelSummary({name: 'Model Summary'}, model);
       }

     /*DOM is a doc object model which treats html/xml doc as a tree structure
     *DOM content loaded event is fired when the document is completly loaded and parsed,without
     *waiting for stylesheets,images and subframes to finish loading.*/
      document.addEventListener('DOMContentLoaded', run);

      //define the model architecture

      function createModel() {
        // Create a sequential model
        //In sequential model the input will flow stright to the o/p,other kinds of model can have branches
        const model = tf.sequential(); 
        
        // Add a single hidden layer
        /* we need to define the input shape because this is the first layer,here input shape is 1
        *because we have 1 number as i/p(horsepower).
        *units sets how big the weight matrix will be in the layer. 
        *By setting it to 1 here we are saying there will be 1 weight 
        *for each of the input features of the data.*/
        model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
        
        // Add an output layer
        model.add(tf.layers.dense({units: 1, useBias: true}));
      
        return model;
      }
      //prepare the data for training
      function convertToTensor(data) {
        // Wrapping these calculations in a tidy will dispose any 
        // intermediate tensors.
        
        //tf.tidy saves memory by removing the intermediate tensors except the tensor it returns
        return tf.tidy(() => {
          // Step 1. Shuffle the data 
          //shuffling helps each have a variety of data across the data distribution.  
          tf.util.shuffle(data);
      
          // Step 2. Convert data to Tensor
          const inputs = data.map(d => d.horsepower)
          const labels = data.map(d => d.mpg);
          
          //the tensors will have a shape of [num_examples,num_features_per_examples]
          //Here we have inputs.length example and each example has 1 input feature(horsepower)
          const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
          const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
      
          //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
          /*for every feauture the min value gets transformed to 0 and the max value gets transformed
          *to 1,and every other value gets transformed into a decimal between 0 and 1*/
          const inputMax = inputTensor.max();
          const inputMin = inputTensor.min();  
          const labelMax = labelTensor.max();
          const labelMin = labelTensor.min();
          /*the internals of many machine learning models will buid with tf.js are 
          *designed to work with numbers that are not too big.so normalize the data between 0-1*/
          const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
          const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));
          /*we want to keep the values used for normalization during training 
          *so we can un normalize the outputs to get them back into original scale*/
          return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            // Return the min/max bounds so we can use them later.
            inputMax,
            inputMin,
            labelMax,
            labelMin,
          }
        });  
      }