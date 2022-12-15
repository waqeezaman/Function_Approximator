import processing.core.*; 
import processing.data.*; 
import processing.event.*; 
import processing.opengl.*; 

import java.util.*; 

import java.util.HashMap; 
import java.util.ArrayList; 
import java.io.File; 
import java.io.BufferedReader; 
import java.io.PrintWriter; 
import java.io.InputStream; 
import java.io.OutputStream; 
import java.io.IOException; 

public class Function_Approximator extends PApplet {


NeuralNetwork SigmoidNetwork = new NeuralNetwork(new Layer[]{
  new DenseLayer(1, 10), 
  new SigmoidLayer(), 
  new DenseLayer(10, 10), 
  new SigmoidLayer(), 
  new DenseLayer(10, 10), 
  new SigmoidLayer(), 
  new DenseLayer(10, 1)

  }, 0.001f);

NeuralNetwork RELUNetwork = new NeuralNetwork(new Layer[]{
  new DenseLayer(1, 10), 
  new RELULayer(), 
  new DenseLayer(10, 10), 
  new RELULayer(), 
  new DenseLayer(10, 10), 
  new RELULayer(), 
  new DenseLayer(10, 10), 
  new RELULayer(), 
  new DenseLayer(10, 10), 
  new RELULayer(), 
  new DenseLayer(10, 10), 
  new RELULayer(), 
  new DenseLayer(10, 10), 
  new RELULayer(), 
  new DenseLayer(10, 10), 
  new RELULayer(), 
  new DenseLayer(10, 1)
  }, 0.001f);


NeuralNetwork TanhNetwork = new NeuralNetwork(new Layer[]{
  new DenseLayer(1, 10), 
  new TanhLayer(), 
  new DenseLayer(10, 10), 
  new TanhLayer(), 
  new DenseLayer(10, 10), 
  new TanhLayer(), 
  new DenseLayer(10, 10), 
  new TanhLayer(), 
  new DenseLayer(10, 10), 
  new TanhLayer(), 
  new DenseLayer(10, 10), 
  new TanhLayer(), 
  new DenseLayer(10, 10), 
  new TanhLayer(), 
  new DenseLayer(10, 10), 
  new TanhLayer(), 
  new DenseLayer(10, 1)
  }, 0.001f);


NeuralNetwork GaussianNetwork = new NeuralNetwork(new Layer[]{
  new DenseLayer(1, 10), 
  new GaussianLayer(), 
  new DenseLayer(10, 10), 
  new GaussianLayer(), 
  new DenseLayer(10, 10), 
  new GaussianLayer(), 
  new DenseLayer(10, 10), 
  new GaussianLayer(), 
  new DenseLayer(10, 10), 
  new GaussianLayer(), 
  new DenseLayer(10, 10), 
  new GaussianLayer(), 
  new DenseLayer(10, 10), 
  new GaussianLayer(), 
  new DenseLayer(10, 10), 
  new GaussianLayer(), 
  new DenseLayer(10, 1)
  }, 0.001f);

NeuralNetwork SoftPlusNetwork = new NeuralNetwork(new Layer[]{
  new DenseLayer(1, 10), 
  new SoftPlusLayer(), 
  new DenseLayer(10, 10), 
  new SoftPlusLayer(), 
  new DenseLayer(10, 10), 
  new SoftPlusLayer(), 
  new DenseLayer(10, 10), 
  new SoftPlusLayer(), 
  new DenseLayer(10, 10), 
  new SoftPlusLayer(), 
  new DenseLayer(10, 10), 
  new SoftPlusLayer(), 
  new DenseLayer(10, 10), 
  new SoftPlusLayer(), 
  new DenseLayer(10, 10), 
  new SoftPlusLayer(), 
  new DenseLayer(10, 1)
  }, 0.001f);






// list of networks
NeuralNetwork[] Networks=new NeuralNetwork[]{RELUNetwork, SigmoidNetwork, TanhNetwork, GaussianNetwork, SoftPlusNetwork};
String[] NetworkNames = new String[]{"RELU", "Sigmoid", "Tanh", "Gaussian", "SoftPlus"};

// index of current network being used
int CurrentNetwork=0;

NeuralNetwork nn= Networks[CurrentNetwork];

// key bindings
char TrainKey = 't';
char ResetKey='r';
char SwitchKey = 's';
char ResetNetworkKey='n';

int BackgroundColour=220;



Graph graph=new Graph(1000, 1000, 10, 10);
int UIWidth =500;
TEXTBOX LearningRateEntry=new TEXTBOX(graph.Width+10, 445, 200, 50);

// matrix holds the points that the user draws on, and which are used to train network
Matrix TrainingPoints=new Matrix();
// matrix holds the points that our network has approximated
Matrix ApproximationPoints=new Matrix();



public void settings() {
  size(graph.Width+UIWidth, graph.Height);
}

public void setup() {
  DrawPointMatrix(TrainingPoints, new Colour(), new Colour(), 0, false, true, 5, false);
  graph.Draw();
}

public void draw() {
 
  nn.LearningRate=GetLearningRate();

  if (TrainingPoints.ColNum>0) {
    for (int i =0; i<125; i++) {
      int randompoint=floor(random(0, TrainingPoints.ColNum)); // get random training point
      float pointx=TrainingPoints.Get(0, randompoint);
      float pointy= TrainingPoints.Get(1, randompoint);
      nn.Train(new Matrix(1, 1, new Float[]{pointx}), new Matrix(1, 1, new Float[]{pointy}));
    }
  }

  DrawAll();
  DrawUI();
}

public Matrix SamplePoints(Matrix points, float start, float end, float density, Function function) {
  // function returns a matrix of points, which sample values from a function, from start to end, with a particular density
  for (float x =start; x<=end; x+=density) {
    points.AddColumn(new Float[]{x, function.function(x)});
  }
  return points;
}

public Matrix SamplePointsFromNetwork( float start, float end, float density, NeuralNetwork Network) {
  // function returns a matrix of points, which samples points from a network
  Matrix points =new Matrix();
  for (float x =start; x<=end; x+=density) {
    points.AddColumn(new Float[]{x, Network.Query(new Matrix(1, 1, new Float[]{x})).Get(0, 0)});
  }
  return points;
}




public void mousePressed() {

  AddPoint();

  LearningRateEntry.PRESSED(mouseX, mouseY);
}

public void keyPressed() {
  if ( key==TrainKey) {
    TrainNetwork();
  } else if ( key== ResetKey) {
    Reset();
  } else if (key==SwitchKey) {
    SwitchNetwork();
  } else if (key==ResetNetworkKey) {
    ResetNetwork();
  }


  LearningRateEntry.KEYPRESSED(key, keyCode);
}



public void AddPoint() {
  if (mouseX>0 && mouseX<graph.Width && mouseY>0 && mouseY<graph.Height) { // is the mouse press within the graph
    for (int col=0; col<TrainingPoints.ColNum; col++) {
      if (TrainingPoints.Get(0, col)==mouseX) {
        // two distinct points cannot have the same x value as this is a function
        return;
      }
    }
    Point newPoint = new Point(mouseX, mouseY);
    
    // map point from canvas to cartesian co-ordinates
    newPoint=MapPointToCartesian(newPoint, graph.ColNum, graph.RowNum, graph.Width, graph.Height);
    TrainingPoints.AddColumn(new Float[]{newPoint.x, newPoint.y});

    // reset the network everytime we add a point
    nn.Reset();

    // resample points from network, which should now be random
    ApproximationPoints = SamplePointsFromNetwork(-5, 5, 0.1f, nn);
  }
}

public void TrainNetwork() {
  // trains the network for 10000 iterations
  for (int i=0; i<10000; i++) {
    float pointx=TrainingPoints.Get(0, i % TrainingPoints.ColNum);
    float pointy= TrainingPoints.Get(1, i% TrainingPoints.ColNum);
    nn.Train(new Matrix(1, 1, new Float[]{pointx}), new Matrix(1, 1, new Float[]{pointy}));
  }
}

public void DrawAll() {
  
  background(BackgroundColour);
  
  //draw graph
  graph.Draw();
  
  // network has been updated so must resample points before drawing
  ApproximationPoints = SamplePointsFromNetwork(-5, 5, 0.1f, nn);
  
  // draw approximation points
  DrawPointMatrix(MapToCanvas(ApproximationPoints, graph.RowNum, graph.ColNum, graph.Width, graph.Height), new Colour(255, 0, 0), new Colour(255, 0, 0), 2, true, true, 0, false);

  //draw training points
  DrawPointMatrix(MapToCanvas(TrainingPoints, graph.RowNum, graph.ColNum, graph.Width, graph.Height), new Colour(), new Colour(), 0, false, true, 15, false);
}

public void Reset() {
  // resets both the points and the network
  ResetPoints();
  ResetNetwork();
  
}

public void ResetPoints() {
  ApproximationPoints=new Matrix();
  TrainingPoints=new Matrix();
}

public void ResetNetwork() {
  nn.Reset();
}

public void SwitchNetwork() {
  CurrentNetwork+=1;
  CurrentNetwork= CurrentNetwork% Networks.length;
  nn=Networks[CurrentNetwork];
  nn.Reset();
}

public void DrawUI() {
  // creates UI box at right hand of screen
  fill(220);
  stroke(0);
  rect(graph.Width, 0, width-graph.Width, height);

  // writes instructions
  stroke(0);
  textFont(createFont("Arial", 20, true), 25);
  fill(0);
  text("Click to add a point", graph.Width+10, 75);
  text("Press R to reset points", graph.Width+10, 150);
  text("Press N to reset network", graph.Width+10, 225);
  text("Press S to switch activation functions", graph.Width+10, 300);
  
  // writes list of different networks
  textFont(createFont("Arial", 20, true), 15);
  String networkslist="";
  for (int i =0; i<NetworkNames.length-1; i++) {
    networkslist+=NetworkNames[i]+" / ";
  }
  networkslist+=NetworkNames[NetworkNames.length-1];
  text(networkslist, graph.Width+10, 330);
  
  // writes current network being used
  text("Current Activation Function: " + NetworkNames[CurrentNetwork], graph.Width+10, 360);

  // draws learning rate entry box
  textFont(createFont("Arial", 20, true), 25);
  text("Learning Rate:", graph.Width+10, 435);
  LearningRateEntry.DRAW();
}

public float GetLearningRate() {
  // returns the learning rate entered in the learning rate entry box
  float lr=PApplet.parseFloat(LearningRateEntry.Text);
  
  if (Float.isNaN(lr)) {
    return 0.0004f;
  }
  return lr;
}

// a class to hold colour values

class Colour{
int R=0;
int G=0; 
int B=0;

  public Colour(int r,int g,int b){
    R=r;
    G=g;
    B=b;
  }
  public Colour(){}

}
// file that holds functions to draw objects to canvas

public void DrawPoint(float x,float y,float radius,Colour colour){
    // draws a point
    stroke(colour.R,colour.G,colour.B);
    fill(colour.R,colour.G,colour.B);
    circle(x,y,radius);
}

public void DrawLine(float x1, float y1,float x2,float y2,Colour colour,float thickness){
  // draws a line
  stroke(colour.R,colour.G,colour.B);
  strokeWeight(thickness);
  line(x1,y1,x2,y2);
}


public void DrawPointMatrix(Matrix points,Colour pointcolour,Colour linecolour,float linethickness,boolean drawline,boolean drawpoints,float PointRadius,boolean cyclical){
  // draws a matrix cointaing a set of points
  
  //draws a line between end and start points
  if(points.ColNum>2 && drawline && cyclical){
    DrawLine(points.Get(0,0),points.Get(1,0),points.Get(0,points.ColNum-1),points.Get(1,points.ColNum-1),
             linecolour,linethickness);
  }
  
  // draws matrix points
  for(int j =0 ;j<points.ColNum;j++){
    
    if(j!=points.ColNum-1 && drawline){ // draws lines between points
      DrawLine(points.Get(0,j),points.Get(1,j),points.Get(0,j+1),points.Get(1,j+1),linecolour,linethickness);
    }
    if(drawpoints){ // draws points
      DrawPoint(points.Get(0,j),points.Get(1,j),PointRadius,pointcolour);

    }
    
    
  }
  

}

public void DrawGrid(int rownum,int colnum,Colour GridLineColour, float GridLinesThickness,int graphwidth,int graphheight){
  
  // draws a grid
  
  
  float xspacing = PApplet.parseFloat(graphwidth/colnum);
  float yspacing = PApplet.parseFloat(graphheight/rownum);
  
  // draw rows
  for(int r =0; r<rownum;r++){
    DrawLine(0,r*yspacing,graphwidth,r*yspacing,GridLineColour,GridLinesThickness);
  }
  //draw columns
  for(int c =0; c<colnum;c++){
    DrawLine(c*xspacing,0,c*xspacing,graphheight,GridLineColour,GridLinesThickness);
  }
  
  // draws axis, with double thickness
  DrawLine(graphwidth/2,0,graphwidth/2,graphheight,GridLineColour,GridLinesThickness*2);
  DrawLine(0,graphheight/2,graphwidth,graphheight/2,GridLineColour,GridLinesThickness*2);
  
  
} 

public Point MapPointToCanvas(Point point,int colnum,int rownum,int graphwidth,int graphheight){
  // maps a point from the cartesian grid to canvas
  float x = point.x+PApplet.parseFloat(colnum/2);
  x*=PApplet.parseFloat(graphheight/colnum);
  float y = point.y*-1;
  y+=PApplet.parseFloat(rownum/2);
  y*=PApplet.parseFloat(graphheight/rownum);
  return new Point(x,y);
}
public Point MapPointToCartesian(Point point,int colnum,int rownum,int graphwidth,int graphheight){
  // maps a single point from the canvas to the cartesian grid
  float x = point.x/PApplet.parseFloat(graphwidth/colnum);
  x-=PApplet.parseFloat(colnum/2);
  
  float y = point.y/ PApplet.parseFloat(graphheight/rownum);
  y*=-1;
  y+=PApplet.parseFloat(rownum/2);
  return new Point(x,y);
}




public Matrix MapToCanvas(Matrix points,int rownum,int colnum,int graphwidth,int graphheight){
  // maps cartesian co-ordinates to canvas co-ordinates
  
  Matrix transformedmatrix= new Matrix(points.RowNum,points.ColNum);
  
   
  transformedmatrix.ApplyBinaryOperationToRow(new AdditionFunc(),0,graphwidth/2);
  transformedmatrix.ApplyBinaryOperationToRow(new AdditionFunc(),1,graphheight/2);
  
  Matrix transformedpoints = new Matrix(points);
  
  transformedpoints.ApplyBinaryOperationToRow(new MultiplicationFunc(),0,graphwidth/rownum);
  transformedpoints.ApplyBinaryOperationToRow(new MultiplicationFunc(),1,-graphheight/colnum);

  transformedmatrix.OperationBetweenMatrices(new AdditionFunc(),transformedpoints);




  
  return transformedmatrix;
}

public Matrix MapToCartesian(Matrix points,int colnum,int rownum,int graphwidth,int graphheight){
  // maps a matrix of points from canvas co-ordinates to cartesian grid co-ordinates
  Matrix transformedmatrix= new Matrix();
  
  float horizontalscale = 1/(PApplet.parseFloat(graphwidth)/PApplet.parseFloat(colnum));
  float verticalscale = 1/(PApplet.parseFloat(graphheight)/PApplet.parseFloat(rownum));

  transformedmatrix=Matrix.ApplyBinaryOperationToRow(new MultiplicationFunc(),points,0,horizontalscale);
  transformedmatrix.ApplyBinaryOperationToRow(new MultiplicationFunc(),0, verticalscale);
  
     // flip row
  transformedmatrix.ApplyBinaryOperationToRow(new MultiplicationFunc(),1, -1f);
  
    transformedmatrix.ApplyBinaryOperationToRow(new AdditionFunc(),0, -PApplet.parseFloat(colnum/2));
  transformedmatrix.ApplyBinaryOperationToRow(new AdditionFunc(),1, PApplet.parseFloat(rownum/2));


  
  return transformedmatrix;
}




class Graph{
  public int Height;
  public int Width;
  private int  ColNum;
  private int RowNum;
  
  private Colour GridLineColour= new Colour(0,0,0);
  private float GridLineThickness= 1;
  
  private Matrix XAxis = new Matrix();
  private Matrix YAxis = new Matrix();
  
  private Colour AxisColour= new Colour(0,0,0);
  private float AxisThickness= 2;
  
  
  
  ArrayList<Matrix> GridMatrices=new ArrayList<Matrix>();
  
  
  public Graph(int h,int w ,int c,int r){
    Height=h;
    Width=w;
    ColNum=c;
    RowNum=r;  
    
    CreateGridMatrices();
    
    XAxis.AddColumn(new Float[]{-0.5f*c,0f});
    XAxis.AddColumn(new Float[]{0.5f*c,0f});
    
    YAxis.AddColumn(new Float[]{0f,-0.5f*r});
    YAxis.AddColumn(new Float[]{0f,0.5f*r});



  }

  
  
  public void CreateGridMatrices() {
  // creates a list of matrices based on the height and width of graph, and how many rows and columns there are 
  GridMatrices= new ArrayList<Matrix>();
  
  // adds rows
  for (float r =-RowNum; r<RowNum; r++) {
    Matrix m=new Matrix(2, 2, new Float[]{-100f, 100f, r, r});

    GridMatrices.add(m);
  }
  
  //adds columns
  for (float c =-ColNum; c<ColNum; c++) {
    Matrix m = new Matrix(2, 2, new Float[]{c, c, -100f, 100f});


    GridMatrices.add(m);
  }
}

public void Draw(){
  // draws graph
  
  // draws list of matrices
  for (Matrix m : GridMatrices){
   DrawPointMatrix(MapToCanvas(m,RowNum,ColNum,Width,Height),GridLineColour,GridLineColour,GridLineThickness,true,false,0,false);
  }
  
  // draws axis
  DrawPointMatrix(MapToCanvas(XAxis,RowNum,ColNum,Width,Height),AxisColour,AxisColour,AxisThickness,true,false,0,false);  
  DrawPointMatrix(MapToCanvas(YAxis,RowNum,ColNum,Width,Height),AxisColour,AxisColour,AxisThickness,true,false,0,false);  

}


}
// an interface holds the template for a function 
// classes implement the actual function or binary function
// instances of these classes can then be passed as parameters 
// for example a method in the matrix class, may take a function as a parameter, and then apply that function on all 
// elements within the matrix 

// the interface for a function
@FunctionalInterface
  public interface Function {
  public float function(Float a);
}

// the interface for a binary function, takes two inputs, returns single output
@FunctionalInterface
  public interface BinaryFunction {
  public float binaryfunction(Float a, Float b);
}


// Activstion functions and their derivatives

class RELUFunc implements Function {
  public float function(Float a) {
    if (a>0) {
      return a;
    } else {
      return 0.01f*a;
    }
  }
}

class RELUDerivFunc implements Function {
  public float function(Float a) {
    if (a>0) {
      return 1;
    } else {
      return 0.01f;
    }
  }
}

class SigmoidFunc implements Function {
  public float function(Float x) {
    return 1/(1+ pow((float)java.lang.Math.E,-x));
  }
}

class SigmoidDerivFunc implements Function {
  public float function(Float x) {
    float sigmoid=1/(1+ pow((float)java.lang.Math.E,-x));
    return sigmoid*(1-sigmoid);
  }
}

class TanhFunc implements Function {
  public float function(Float x) {
    return (float)java.lang.Math.tanh(x);
  }
}

class TanhDerivFunc implements Function {
  public float function(Float x) {
    float tanh= (float)java.lang.Math.tanh(x);
    return 1- tanh*tanh;
  }
}

class GaussianFunc implements Function {
  public float function(Float x) {
    return pow((float)java.lang.Math.E,-(x*x));
  }
}

class GaussianDerivFunc implements Function {
  public float function(Float x) {
    return -2*x*pow((float)java.lang.Math.E,-(x*x));
  }
}

class SoftPlusFunc implements Function {
  public float function(Float x) {
    return (float)java.lang.Math.log(1+pow((float)java.lang.Math.E,x));
  }
}

class SoftPlusDerivFunc implements Function {
  public float function(Float x) {
    return 1/(1+pow((float)java.lang.Math.E,-x));
  }
}

class SinFunc implements Function {
  public float function(Float x) {
    return sin(x);
  }
}


//// other functions

class RandomFunc implements BinaryFunction {
  public float binaryfunction(Float a, Float b) {
    return random(a, b);
  }
}

class AdditionFunc implements BinaryFunction {
  public float binaryfunction(Float a, Float b) {
    return a+b;
  }
}

class SubtractionFunc implements BinaryFunction {
  public float binaryfunction(Float a, Float b) {
    return a-b;
  }
}


class MultiplicationFunc implements BinaryFunction {
  public float binaryfunction(Float a, Float b) {
    return a*b;
  }
}

class PowerFunc implements BinaryFunction {
  public float binaryfunction(Float a, Float power) {
    return pow(a, power);
  }
}
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

  public void OutputLayer() {
    // outputs layer
    Input.OutputMatrix("Input");
    Output.OutputMatrix("Output");
    Weights.OutputMatrix("Weights");
    Bias.OutputMatrix("Bias");
  }
  
  public void Reset(){
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
  
  public void OutputLayer() {
    Input.OutputMatrix("Input");
    Output.OutputMatrix("Output");
  }
  
  public void Reset(){}
  
  
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
  
  public void OutputLayer() {
    Input.OutputMatrix("Input");
    Output.OutputMatrix("Output");
  }
  
  public void Reset(){}
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
  
  public void OutputLayer() {
    Input.OutputMatrix("Input");
    Output.OutputMatrix("Output");
  }
  
  public void Reset(){}
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
  
  public void OutputLayer() {
    Input.OutputMatrix("Input");
    Output.OutputMatrix("Output");
  }
  
  public void Reset(){}
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
  
  public void OutputLayer() {
    Input.OutputMatrix("Input");
    Output.OutputMatrix("Output");
  }
  
  public void Reset(){}
}
 static class Matrix {

  int RowNum;
  int ColNum;

  Float[][] matrix;


  public Matrix() {
    // creates matrix object with nothing in it
  }
  public Matrix(Matrix copy) {
    // takes a matrix, and creates a new matrix with the exact same dimensions, and the same values
    // at each index
    CreateZeroMatrix(copy.RowNum, copy.ColNum);
    for (int j=0; j<ColNum; j++) {
      for (int i =0; i<RowNum; i++) {
        Set(i, j, copy.Get(i, j));
      }
    }
  }

  public Matrix(int rownum, int colnum) {
    //creates an zero matrix of rownum by colnum
    RowNum=rownum;
    ColNum=colnum;
    CreateZeroMatrix(rownum, colnum);
  }

  public Matrix(int rownum, int colnum, Float[] values) {
    // creates matrix of rownum by colnum and fills with values from array
    // length of array of values passed must be less than size of matrix
    // if the length of the array of values is smaller than the size of the matrix, then the remaining values are set to zero
    // values must be in row order
    // e.g if i have a 2x3 matrix, and i pass in an array of values,
    // the first two elements of the array would correspond to the first row of the matrix
    
    CreateZeroMatrix(rownum, colnum);
    for (int i=0; i<values.length; i++) {
      Set(i/colnum, Math.floorMod(i, colnum), values[i]);
    }
  }

  private  void CreateZeroMatrix(int rownum, int colnum) {
    // creates a matrix of size rownum by colnum, and sets all values to 0
    matrix=new Float[rownum][colnum];
    RowNum=rownum;
    ColNum=colnum;
    for (int i =0; i<rownum; i++) {
      for (int j =0; j<colnum; j++) {
        Set(i, j, 0F);
      }
    }
  }

  private void CopyMatrix(Matrix B) {
    //copies contents of B onto original, leaves any extra columns or rows as they are
    if (ColNum>=B.ColNum && RowNum>=B.RowNum) {
      for (int i=0; i<B.RowNum; i++) {
        for (int j=0; j<B.ColNum; j++) {
          Set(i, j, B.Get(i, j));
        }
      }
    }
  }






  public void ApplyOperationToRow(Function func, int row) {
    // applies a function to every element in a given row of matrix
    for (int j=0; j<ColNum; j++) {
      Set(row, j, func.function(Get(row, j)));
    }
  }
  
  public void ApplyOperationToColumn(Function func, int col) {
     // applies a function to every element in a given column of matrix
    for (int i=0; i<RowNum; i++) {
      Set(i, col, func.function(Get(i, col)));
    }
  } 



  public void ApplyBinaryOperationToRow(BinaryFunction func, int row, float value) {
    // applies a binary function to every element in a given row of matrix

    for (int j=0; j<ColNum; j++) {
      Set(row, j, func.binaryfunction(Get(row, j), value));
    }

  }
  public void ApplyBinaryOperationToColumn(BinaryFunction func, int col, float value) {
    // applies a binary function to every element in a given column of matrix
    for (int i=0; i<RowNum; i++) {
      Set(i, col, func.binaryfunction(Get(i, col), value));
    }
  } 




  public void SetColumn(int col, Float[] colvalues) {
    // sets a column to the values in the array even if the array is smaller than the column, but not if the array is larger than the column
    // if the array is smaller, then values at indexes larger than the largest index in the array are left unchanged
    
    if (col<0 || col>ColNum) { // checks if column exists
      println("SetColumn()  ---ERROR--- referencing column that doesnt exist");
      return;
    }
    if (colvalues.length<=RowNum) { // checks array is smaller than size of column
      for (int r=0; r<colvalues.length; r++) {
        Set(r, col, colvalues[r]);
      }
    } else {
      println("Tried to set a column to a set of values which was larger than the column ");
    }
  }

  public void SetRow(int row, Float[] rowvalues) {
    // sets a row to the values in the array even if the array is smaller than the row, but not if the array is larger than the row
    // if the array is smaller, then values at indexes larger than the largest index in the array are left unchanged
    if (row<0 || row>RowNum) { // checks row exists
      println("SetRow() ---ERROR--- referencing row that doesnt exist");
      return;
    }
    if (rowvalues.length<=ColNum) {  // checks array is smaller than size of row
      for (int c=0; c<rowvalues.length; c++) {
        Set(row, c, rowvalues[c]);
      }
    } else {
      println("SetRow() ---ERROR--- Tried to set a row to a set of values which was larger than the row ");
      return;
    }
  }

  public static Matrix Multiply(Matrix A, Matrix B) {  // multiplies two matrices together
    if (B.ColNum!=A.RowNum) {  // checks matrices can be multiplied
      println("Matrix Multiply() ---ERROR--- trying to multiply matrices where sizes don't correspond");
      println("Size of A:" + A.RowNum + ":"+A.ColNum);
      println("Size of B:" + B.RowNum + ":"+B.ColNum);
      return null;
    } else {
      
      Matrix transformedmatrix= new Matrix(B.RowNum, A.ColNum);


      for (int row=0; row<B.RowNum; row++) {
        for (int col=0; col<A.ColNum; col++) {
          for (int common=0; common<B.ColNum; common++) {
            float sum=transformedmatrix.Get(row, col);
            float a = B.Get(row, common);
            float b = A.Get(common, col);
            transformedmatrix.Set(row, col, sum+a*b);
          }
        }
      }
      return transformedmatrix;
    }
  }

  public void Set(int row, int col, Float val) {  
    matrix[row][col]=val;
  }

  public Float Get(int row, int col) {
    return  matrix[row][col];
  }



  private void AddColumn() {
    // adds a column full of zeros to a matrix
    if(RowNum==0){
      println("AddColumn() --- ERROR -- Tried to add a column to a matrix with zero rows, must add column with a set of a set of values to instantiate rows");
      return;
    }
    Matrix oldmatrix=new Matrix(this);
    ColNum+=1;
    CreateZeroMatrix(RowNum, ColNum);
    CopyMatrix(oldmatrix);
  }

  private void AddRow() {
    // adds a row full of zeros to a matrix
    Matrix oldmatrix=new Matrix(this);
    RowNum+=1;
    CreateZeroMatrix(RowNum, ColNum);
    CopyMatrix(oldmatrix);
  }


  public void AddColumn(Float[] values) {
    // adds a column of values to matrix
    if (RowNum==0) {
      RowNum=values.length;
    }
    AddColumn();
    SetColumn(ColNum-1, values);
  }

  public void AddRow(Float[] values) {
    // adds a row of values to a matrix
    if (ColNum==0) {
      ColNum=values.length;
    }
    AddRow();
    SetRow(RowNum-1, values);
  }

  public Matrix Transpose() {
    // returns transpose of matrix
    Matrix transpose=new Matrix();
    for (int c=0; c<RowNum; c++) {
      transpose.AddColumn(GetRow(c));
    }

    return transpose;
  }




  public Float[] GetRow(int row) {
    // returns entire row of matrix
    if (row>RowNum || row<0) {
      println("GetRow() ---ERROR--- tried to get row that doesnt exist");
      return null;
    }
    return matrix[row];
  }
  public Float[] GetColumn(int col) {
    // returns entire column of matrix
    if (col>RowNum || col<0) {
      println("tried to get column that doesnt exist");
      return null;
    }
    Float[] colarr =new Float[RowNum];
    for (int r=0; r<RowNum; r++) {
      colarr[r]=Get(r, col);
    }
    return colarr;
  }


  public void OutputColumn(int col) {
    println("Column: " + col);
    println(GetColumn(col));
  }
  public void OutputRow(int row) {
    println("Row: "+ row);
    println(GetRow(row));
  }

  public void OutputMatrix() {
    println("--------------------");
    for (int i =0; i<RowNum; i++) {
      String row="[";
      for (int j =0; j<ColNum; j++) {
        row=row + Get(i, j) +" , ";
      }
      println(row +"]");
    }
    println("--------------------");
  }
  
    public void OutputMatrix(String description) {
    println("--------- " + description + "-----------");
    for (int i =0; i<RowNum; i++) {
      String row="[";
      for (int j =0; j<ColNum; j++) {
        row=row + Get(i, j) +" , ";
      }
      println(row +"]");
    }
    println("--------------------");
  }


  public void ApplyFunction(Function func) {
    // applies a function to every element in the matrix
    for (int i=0; i<RowNum; i++) {
      for (int j=0; j<ColNum; j++) {
        Set(i, j, func.function(Get(i, j)));
      }
    }
  }
  public void ApplyBinaryFunction(BinaryFunction func, float value) {
    // applies a binary function to every element in the matrix, with value provided
    for (int i=0; i<RowNum; i++) {
      for (int j=0; j<ColNum; j++) {
        Set(i, j, func.binaryfunction( Get(i, j), value));
      }
    }
  }

  public void ApplyBinaryFunction(BinaryFunction func, float value1, float value2) {
    // sets every element in the matrix equal to the value returned by applying the binary function given 
    // to value and value2
    for (int i=0; i<RowNum; i++) {
      for (int j=0; j<ColNum; j++) {
        Set(i, j, func.binaryfunction( value1, value2));
      }
    }
  }

  public void OperationBetweenMatrices( BinaryFunction func,Matrix B) {
    // applies a binary function between this matrix, and matrix B element-wise
    if (RowNum!=B.RowNum || ColNum!= B.ColNum) {
      println("ERROR: tried to apply binary operation between two matrices of different sizes");
      return;
    }

    for (int i =0; i<RowNum; i++) {
      for (int j =0; j<ColNum; j++) {
        Set(i, j, func.binaryfunction(Get(i, j), B.Get(i, j)));
      }
    }
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // STATIC METHODS
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  public static Matrix ApplyFunction(Function func, Matrix A) {
    // applies a function to every element of matrix A
    Matrix newMatrix=new Matrix(A.RowNum, A.ColNum);
    for (int i=0; i<A.RowNum; i++) {
      for (int j=0; j<A.ColNum; j++) {
        newMatrix.Set(i, j, func.function(A.Get(i, j)));
      }
    }
    return newMatrix;
  }

  public static Matrix ApplyBinaryFunction(BinaryFunction func, Matrix A, float value1, float value2) {
    // sets every element in matrix A, to the value returned by the binary function, when it takes 
    // value1 and value 2 as arguments
    Matrix newMatrix =new Matrix(A.RowNum, A.ColNum);
    for (int i=0; i<A.RowNum; i++) {
      for (int j=0; j<A.ColNum; j++) {
        newMatrix.Set(i, j, func.binaryfunction( value1, value2));
      }
    }
    return newMatrix;
  }
  
  public static Matrix ApplyBinaryFunction(BinaryFunction func, Matrix A, float value) {
    // applies a binary function to every element of matrix A, and the value given
    Matrix newMatrix=new Matrix(A.RowNum, A.ColNum);
    for (int i=0; i<A.RowNum; i++) {
      for (int j=0; j<A.ColNum; j++) {
        newMatrix.Set(i, j, func.binaryfunction( A.Get(i, j), value));
      }
    }
    return newMatrix;
  }

  public static Matrix OperationBetweenMatrices(BinaryFunction func, Matrix A, Matrix B) {
    // applies a binary function to two matrices of the same size, element-wise
    if (A.RowNum!=B.RowNum || A.ColNum!= B.ColNum) {
      println("static Matrix OperationBetweenMatrices() ---ERROR--- tried to apply binary operation between two matrices of different sizes");
      return null;
    }

    Matrix newMatrix = new Matrix(B.RowNum, B.ColNum);
    for (int i =0; i<A.RowNum; i++) {
      for (int j =0; j<A.ColNum; j++) {
        newMatrix.Set(i, j, func.binaryfunction(A.Get(i, j), B.Get(i, j)));
      }
    }
    return newMatrix;
  }
  
  
  public static Matrix ApplyOperationToRow(Function func, Matrix A,int row) {
    // applies a functuion to every element in a row 
    Matrix newMatrix=new Matrix(A.RowNum, A.ColNum);
    for (int j=0; j<A.ColNum; j++) {
      newMatrix.Set(row, j, func.function(A.Get(row, j)));
    }
    return newMatrix;
  }
  public static Matrix ApplyOperationToColumn(Function func, Matrix A,int col) {
    // applies a functuion to every element in a column

    Matrix newMatrix=new Matrix(A.RowNum, A.ColNum);
    for (int i=0; i<A.RowNum; i++) {
      newMatrix.Set(i, col, func.function(A.Get(i, col)));
    }
    return newMatrix;
  } 
  
  
  public static Matrix ApplyBinaryOperationToRow(BinaryFunction func, Matrix A,int row,float x) {
    // applies a binary function to every element in a row, and x

    Matrix newMatrix=new Matrix(A.RowNum, A.ColNum);
    for (int j=0; j<A.ColNum; j++) {
      newMatrix.Set(row, j, func.binaryfunction(A.Get(row, j),x));
    }
    return newMatrix;
  }
  public static Matrix ApplyBinaryOperationToColumn(BinaryFunction func, Matrix A,int col,float x) {
    // applies a binary function to every element in a column, and x
    Matrix newMatrix=new Matrix(A.RowNum, A.ColNum);
    for (int i=0; i<A.RowNum; i++) {
      newMatrix.Set(i, col, func.binaryfunction(A.Get(i, col),x));
    }
    return newMatrix;
  } 

}
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
 
// class for point objects

class Point{
  public float x;
  public float y; 
  

  public  Point(float xpos, float ypos){
    x=xpos;
    y=ypos;
  }
}
// code for this class from https://github.com/mitkonikov/Processing/blob/master/Text_Box/TEXTBOX.pde

public class TEXTBOX {
   public int X = 0, Y = 0, H = 35, W = 200;
   public int TEXTSIZE = 40;
   
   // COLORS
   public int Background = color(175, 175, 175);
   public int Foreground = color(0, 0, 0);
   public int BackgroundSelected = color(190, 190, 190);
   public int Border = color(30, 30, 30);
   
   public boolean BorderEnable = false;
   public int BorderWeight = 1;
   
   public String Text = "";
   public int TextLength = 0;

   private boolean selected = false;
   
   TEXTBOX() {
      // CREATE OBJECT DEFAULT TEXTBOX
   }
   
   TEXTBOX(int x, int y, int w, int h) {
      X = x; Y = y; W = w; H = h;
   }
   
   public void DRAW() {
      // DRAWING THE BACKGROUND
      if (selected) {
         fill(BackgroundSelected);
      } else {
         fill(Background);
      }
      
      if (BorderEnable) {
         strokeWeight(BorderWeight);
         stroke(Border);
      } else {
         noStroke();
      }
      
      rect(X, Y, W, H);
      
      // DRAWING THE TEXT ITSELF
      fill(Foreground);
      textSize(TEXTSIZE);
      text(Text, X + (textWidth("a") / 2), Y + TEXTSIZE);
   }
   
   // IF THE KEYCODE IS ENTER RETURN 1
   // ELSE RETURN 0
   public boolean KEYPRESSED(char KEY, int KEYCODE) {
      if (selected) {
         if (KEYCODE == (int)BACKSPACE) {
            BACKSPACE();
         } else if (KEYCODE == 32) {
            // SPACE
            addText(' ');
         } else if (KEYCODE == (int)ENTER) {
            return true;
         }
         else {
            // CHECK IF THE KEY IS A LETTER OR A NUMBER OR COMMA
            boolean isKeyCapitalLetter = (KEY >= 'A' && KEY <= 'Z');
            boolean isKeySmallLetter = (KEY >= 'a' && KEY <= 'z');
            boolean isKeyNumber = (KEY >= '0' && KEY <= '9');
            boolean isComma = (KEY ==',');
            boolean isFullStop=(KEY=='.');
            boolean isMinus=(KEY=='-');
      
          //  if (isKeyCapitalLetter || isKeySmallLetter || isKeyNumber || isComma || isFullStop) {
            if(isKeyNumber || isFullStop || isMinus){
               addText(KEY);
            }
         }
      }
  
      
      return false;
   }
   
   private void addText(char text) {
      // IF THE TEXT WIDHT IS IN BOUNDARIES OF THE TEXTBOX
      if (textWidth(Text + text) < W) {
         Text += text;
         TextLength++;
      }
   }
   
   private void BACKSPACE() {
      if (TextLength - 1 >= 0) {
         Text = Text.substring(0, TextLength - 1);
         TextLength--;
      }
   }
   
   // FUNCTION FOR TESTING IS THE POINT
   // OVER THE TEXTBOX
   private boolean overBox(int x, int y) {
      if (x >= X && x <= X + W) {
         if (y >= Y && y <= Y + H) {
            return true;
         }
      }
      
      return false;
   }
   
   public void PRESSED(int x, int y) {
      if (overBox(x, y)) {
         selected = true;
      } else {
         selected = false;
      }
   }
}
  static public void main(String[] passedArgs) {
    String[] appletArgs = new String[] { "Function_Approximator" };
    if (passedArgs != null) {
      PApplet.main(concat(appletArgs, passedArgs));
    } else {
      PApplet.main(appletArgs);
    }
  }
}
