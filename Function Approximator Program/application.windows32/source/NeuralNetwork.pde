public class NeuralNetwork {

  public Layer[] Layers;
  float LearningRate;
  float Error;
  
  public NeuralNetwork(Layer[] layers, float lr) {
    LearningRate=lr;
    Layers=layers;
  }

  public Matrix Query(Matrix inputs) {
    FeedForward(inputs);
    return Layers[Layers.length-1].Output;
  }

  private void FeedForward(Matrix inputs) {
    Layers[0].FeedForward(inputs);
    for (int i =1; i<Layers.length; i++) {
      Layers[i].FeedForward(Layers[i-1].Output);
    }
  }

  public void Train(Matrix input, Matrix Target) {
    
    FeedForward(input);
  
    Error+=GetError(Target,Layers[Layers.length-1].Output); // cumulative error

    BackPropogate(GetErrorDeriv(Target,Layers[Layers.length-1].Output));
  }

  private void BackPropogate(Matrix Deriv) {
    Layers[Layers.length-1].BackPropogation(Deriv, LearningRate);
    for (int i =Layers.length-2; i>=0; i--) {
      Layers[i].BackPropogation(Layers[i+1].Error_Input_Deriv, LearningRate);
    }
  }

  private float GetError(Matrix Target, Matrix Output) {
    // returns mean squared error
    Matrix SquareDifference=Matrix.OperationBetweenMatrices(new SubtractionFunc(), Target, Output);
    SquareDifference.ApplyBinaryFunction(new PowerFunc(), 2);
    float sum=0;
    for (int i =0; i<SquareDifference.RowNum; i++) {
      for (int j =0; j<SquareDifference.ColNum; j++) {
        
        sum+=SquareDifference.Get(i,j);
      }
    }
    return sum/SquareDifference.RowNum;
  }
  
  private Matrix GetErrorDeriv(Matrix Target,Matrix Output){
    // returns derivative of mean squared error
    Matrix deriv=Matrix.OperationBetweenMatrices(new SubtractionFunc(),Target,Output);
    deriv.ApplyBinaryFunction(new MultiplicationFunc(),2/Target.RowNum);
    return deriv;
  }
  
  public void OutputNetwork(){
    for(int l =0;l<Layers.length;l++){
      println("Layer Number: "+l);
      Layers[l].OutputLayer();
    }
  }
  
  
  public void Reset(){
    // resets all parameters in network
    Error=0;
    for(Layer l: Layers){
      l.Reset();
    }
  }
}
 
