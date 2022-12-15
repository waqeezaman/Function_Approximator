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
