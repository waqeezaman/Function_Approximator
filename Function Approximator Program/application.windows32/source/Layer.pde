// an abstract class defines the layer 
// all layers inherit from this abstaract class
// a layer must feed forward and backpropogate 

abstract class Layer {
  
  public Matrix Input;  // need to store inputs for backpropogation
  public Matrix Output;

  public Matrix Error_Output_Deriv;
  public Matrix Error_Input_Deriv;

  public abstract void FeedForward(Matrix inputs);
  
  public abstract void BackPropogation(Matrix Error_Output_Deriv, float LearningRate);

  public abstract void OutputLayer();
  
  public abstract void Reset();

}

// dense layer class
class DenseLayer extends Layer {


  Matrix Weights;
  Matrix Bias;

  public DenseLayer(int inputsize, int outputsize) {
    // create weight matrix
    Weights=new Matrix(outputsize, inputsize);
    // create bias matrix
    Bias=new Matrix(outputsize, 1);
    // randomise weights and biases
    Reset();
  }


  public void FeedForward(Matrix input) {

    Input=input;  // input is stored, this is to backpropogate later

    Output=Matrix.Multiply(Input, Weights);  // inputs are multiplied by the weights

    Output.OperationBetweenMatrices(new AdditionFunc(), Bias); // the bias is added on 
  }

  public void BackPropogation(Matrix error_output_deriv, float LearningRate) {

    // calculate the derivative of the error with respect to the weights
    Error_Output_Deriv=  error_output_deriv;

    // calculate derivative of error with respect to inputs
    Error_Input_Deriv= Matrix.Multiply(Error_Output_Deriv, Weights.Transpose());

    // calcualte derivative of error with respect to weights 
    Matrix Weights_Output_Deriv = Matrix.Multiply(Input.Transpose(), Error_Output_Deriv);


    // update  weights 
    Weights.OperationBetweenMatrices(new SubtractionFunc(), Matrix.ApplyBinaryFunction(new MultiplicationFunc(), Weights_Output_Deriv, -LearningRate));

    // derivative of bias is just the derivative of the error to output 
    // update bias
    Bias.OperationBetweenMatrices(new SubtractionFunc(), Matrix.ApplyBinaryFunction(new MultiplicationFunc(), Error_Output_Deriv, -LearningRate));
  }

  void OutputLayer() {
    // outputs layer
    Input.OutputMatrix("Input");
    Output.OutputMatrix("Output");
    Weights.OutputMatrix("Weights");
    Bias.OutputMatrix("Bias");
  }
  
  void Reset(){
      // randomises the weights and biases
      Weights.ApplyBinaryFunction(new RandomFunc(), -1, 1);
      Bias.ApplyBinaryFunction(new RandomFunc(), -1, 1);
  }
}



class RELULayer extends Layer {

  public void FeedForward (Matrix input) {
    Input=input;
    Output=Matrix.ApplyFunction(new RELUFunc(), input);
  }
  
  public void BackPropogation (Matrix error_output_deriv, float LearningRate) {
    
    Error_Output_Deriv=new Matrix(error_output_deriv);
    
    Error_Input_Deriv=Matrix.ApplyFunction(new RELUDerivFunc(), Input);

    Error_Input_Deriv.OperationBetweenMatrices(new MultiplicationFunc(), Error_Output_Deriv);
  }
  
  void OutputLayer() {
    Input.OutputMatrix("Input");
    Output.OutputMatrix("Output");
  }
  
  void Reset(){}
  
  
}

class SigmoidLayer extends Layer {
  
  public void FeedForward(Matrix input) {
    Input=input;
    Output=Matrix.ApplyFunction(new SigmoidFunc(), input);
  }
  
  public void BackPropogation(Matrix error_output_deriv, float LearningRate) {
    Error_Output_Deriv=new Matrix(error_output_deriv);
    Error_Input_Deriv=Matrix.ApplyFunction(new SigmoidDerivFunc(), Input);
    Error_Input_Deriv=Matrix.OperationBetweenMatrices(new MultiplicationFunc(), Error_Input_Deriv, Error_Output_Deriv);
  }
  
  void OutputLayer() {
    Input.OutputMatrix("Input");
    Output.OutputMatrix("Output");
  }
  
  void Reset(){}
}


class TanhLayer extends Layer {
  
  public void FeedForward(Matrix input) {
    Input=input;
    Output=Matrix.ApplyFunction(new TanhFunc(), input);
  }
  
  public void BackPropogation(Matrix error_output_deriv, float LearningRate) {
    Error_Output_Deriv=new Matrix(error_output_deriv);
    Error_Input_Deriv=Matrix.ApplyFunction(new TanhDerivFunc(), Input);
    Error_Input_Deriv=Matrix.OperationBetweenMatrices(new MultiplicationFunc(), Error_Input_Deriv, Error_Output_Deriv);
  }
  
  void OutputLayer() {
    Input.OutputMatrix("Input");
    Output.OutputMatrix("Output");
  }
  
  void Reset(){}
}


class GaussianLayer extends Layer {
  
  public void FeedForward(Matrix input) {
    Input=input;
    Output=Matrix.ApplyFunction(new GaussianFunc(), input);
  }
  
  public void BackPropogation(Matrix error_output_deriv, float LearningRate) {
    Error_Output_Deriv=new Matrix(error_output_deriv);
    Error_Input_Deriv=Matrix.ApplyFunction(new GaussianDerivFunc(), Input);
    Error_Input_Deriv=Matrix.OperationBetweenMatrices(new MultiplicationFunc(), Error_Input_Deriv, Error_Output_Deriv);
  }
  
  void OutputLayer() {
    Input.OutputMatrix("Input");
    Output.OutputMatrix("Output");
  }
  
  void Reset(){}
}


class SoftPlusLayer extends Layer {
  
  public void FeedForward(Matrix input) {
    Input=input;
    Output=Matrix.ApplyFunction(new SoftPlusFunc(), input);
  }
  
  public void BackPropogation(Matrix error_output_deriv, float LearningRate) {
    Error_Output_Deriv=new Matrix(error_output_deriv);
    Error_Input_Deriv=Matrix.ApplyFunction(new SoftPlusDerivFunc(), Input);
    Error_Input_Deriv=Matrix.OperationBetweenMatrices(new MultiplicationFunc(), Error_Input_Deriv, Error_Output_Deriv);
  }
  
  void OutputLayer() {
    Input.OutputMatrix("Input");
    Output.OutputMatrix("Output");
  }
  
  void Reset(){}
}
