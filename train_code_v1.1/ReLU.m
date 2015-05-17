function g = ReLU(z)
%   J = ReLU(z) computes the ReLU activation function of z.
    
    index = (z < 0);
    
    z(index) = 0;       % all numbers of z less than 0 should be cut off
    
    g = z;              % assign z to g
    
end

