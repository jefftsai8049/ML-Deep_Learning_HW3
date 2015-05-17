function y = Softmax(z)
      
    exp_z = exp(z);     % exponential values of each z
    exp_z_sum = sum(exp_z);     % sum of exp_z along comlumn
    y = exp_z / exp_z_sum;      % y should be values between 0~1
    
end