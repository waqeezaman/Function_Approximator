// file that holds functions to draw objects to canvas

void DrawPoint(float x,float y,float radius,Colour colour){
    // draws a point
    stroke(colour.R,colour.G,colour.B);
    fill(colour.R,colour.G,colour.B);
    circle(x,y,radius);
}

void DrawLine(float x1, float y1,float x2,float y2,Colour colour,float thickness){
  // draws a line
  stroke(colour.R,colour.G,colour.B);
  strokeWeight(thickness);
  line(x1,y1,x2,y2);
}


void DrawPointMatrix(Matrix points,Colour pointcolour,Colour linecolour,float linethickness,boolean drawline,boolean drawpoints,float PointRadius,boolean cyclical){
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

void DrawGrid(int rownum,int colnum,Colour GridLineColour, float GridLinesThickness,int graphwidth,int graphheight){
  
  // draws a grid
  
  
  float xspacing = float(graphwidth/colnum);
  float yspacing = float(graphheight/rownum);
  
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

Point MapPointToCanvas(Point point,int colnum,int rownum,int graphwidth,int graphheight){
  // maps a point from the cartesian grid to canvas
  float x = point.x+float(colnum/2);
  x*=float(graphheight/colnum);
  float y = point.y*-1;
  y+=float(rownum/2);
  y*=float(graphheight/rownum);
  return new Point(x,y);
}
Point MapPointToCartesian(Point point,int colnum,int rownum,int graphwidth,int graphheight){
  // maps a single point from the canvas to the cartesian grid
  float x = point.x/float(graphwidth/colnum);
  x-=float(colnum/2);
  
  float y = point.y/ float(graphheight/rownum);
  y*=-1;
  y+=float(rownum/2);
  return new Point(x,y);
}




Matrix MapToCanvas(Matrix points,int rownum,int colnum,int graphwidth,int graphheight){
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

Matrix MapToCartesian(Matrix points,int colnum,int rownum,int graphwidth,int graphheight){
  // maps a matrix of points from canvas co-ordinates to cartesian grid co-ordinates
  Matrix transformedmatrix= new Matrix();
  
  float horizontalscale = 1/(float(graphwidth)/float(colnum));
  float verticalscale = 1/(float(graphheight)/float(rownum));

  transformedmatrix=Matrix.ApplyBinaryOperationToRow(new MultiplicationFunc(),points,0,horizontalscale);
  transformedmatrix.ApplyBinaryOperationToRow(new MultiplicationFunc(),0, verticalscale);
  
     // flip row
  transformedmatrix.ApplyBinaryOperationToRow(new MultiplicationFunc(),1, -1f);
  
    transformedmatrix.ApplyBinaryOperationToRow(new AdditionFunc(),0, -float(colnum/2));
  transformedmatrix.ApplyBinaryOperationToRow(new AdditionFunc(),1, float(rownum/2));


  
  return transformedmatrix;
}
