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

  
  
  void CreateGridMatrices() {
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
