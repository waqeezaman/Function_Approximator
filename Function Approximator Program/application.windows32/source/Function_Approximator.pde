import java.util.*;
NeuralNetwork SigmoidNetwork = new NeuralNetwork(new Layer[]{
  new DenseLayer(1, 10), 
  new SigmoidLayer(), 
  new DenseLayer(10, 10), 
  new SigmoidLayer(), 
  new DenseLayer(10, 10), 
  new SigmoidLayer(), 
  new DenseLayer(10, 1)

  }, 0.001);

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
  }, 0.001);


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
  }, 0.001);


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
  }, 0.001);

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
  }, 0.001);






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



void settings() {
  size(graph.Width+UIWidth, graph.Height);
}

void setup() {
  DrawPointMatrix(TrainingPoints, new Colour(), new Colour(), 0, false, true, 5, false);
  graph.Draw();
}

void draw() {
 
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

Matrix SamplePoints(Matrix points, float start, float end, float density, Function function) {
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




void mousePressed() {

  AddPoint();

  LearningRateEntry.PRESSED(mouseX, mouseY);
}

void keyPressed() {
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



void AddPoint() {
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
    ApproximationPoints = SamplePointsFromNetwork(-5, 5, 0.1, nn);
  }
}

void TrainNetwork() {
  // trains the network for 10000 iterations
  for (int i=0; i<10000; i++) {
    float pointx=TrainingPoints.Get(0, i % TrainingPoints.ColNum);
    float pointy= TrainingPoints.Get(1, i% TrainingPoints.ColNum);
    nn.Train(new Matrix(1, 1, new Float[]{pointx}), new Matrix(1, 1, new Float[]{pointy}));
  }
}

void DrawAll() {
  
  background(BackgroundColour);
  
  //draw graph
  graph.Draw();
  
  // network has been updated so must resample points before drawing
  ApproximationPoints = SamplePointsFromNetwork(-5, 5, 0.1, nn);
  
  // draw approximation points
  DrawPointMatrix(MapToCanvas(ApproximationPoints, graph.RowNum, graph.ColNum, graph.Width, graph.Height), new Colour(255, 0, 0), new Colour(255, 0, 0), 2, true, true, 0, false);

  //draw training points
  DrawPointMatrix(MapToCanvas(TrainingPoints, graph.RowNum, graph.ColNum, graph.Width, graph.Height), new Colour(), new Colour(), 0, false, true, 15, false);
}

void Reset() {
  // resets both the points and the network
  ResetPoints();
  ResetNetwork();
  
}

void ResetPoints() {
  ApproximationPoints=new Matrix();
  TrainingPoints=new Matrix();
}

void ResetNetwork() {
  nn.Reset();
}

void SwitchNetwork() {
  CurrentNetwork+=1;
  CurrentNetwork= CurrentNetwork% Networks.length;
  nn=Networks[CurrentNetwork];
  nn.Reset();
}

void DrawUI() {
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

float GetLearningRate() {
  // returns the learning rate entered in the learning rate entry box
  float lr=float(LearningRateEntry.Text);
  
  if (Float.isNaN(lr)) {
    return 0.0004;
  }
  return lr;
}
