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
      return 0.01*a;
    }
  }
}

class RELUDerivFunc implements Function {
  public float function(Float a) {
    if (a>0) {
      return 1;
    } else {
      return 0.01;
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
