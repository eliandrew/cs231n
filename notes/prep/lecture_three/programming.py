import math
# backprop example for f(x,y) = (x + sigmoid(y)) / (sigmoid(x) + (x+y)^2)

# forward pass

x = 3
y = -4

sigy = 1 / (1 + math.exp(-y))
num = x + sigy

sigx = 1 / (1 + math.exp(-x))
xpy = x + y

xpysqr = xpy**2
den = sigx + xpysqr

invden = 1 / den
f = num * invden

# backward pass f = num * invden

dnum = invden
dinvden = num

# back invden = 1 / den
dden = (-1 / (den**2)) * dinvden

# back den = sigx + xpysqr
dsigx = (1) * dden
dxpysqr = (1) * dden

# back xpysqr = xpy**2
dxpy = (2 * xpy) * dxpysqr

# back xpy = x + y 
dx = (1) * dxpy
dy = (1) * dxpy

# back sigx = 1 / (1 + math.exp(-x))
dx += ((1 - sigx)*sigx) * dsigx

# back num = x + sigy
dx += (1) * dnum
dsigy = (1) * dnum

# back sigy = 1 / (1 + math.exp(-y))
dy += ((1 - sigy)*sigy) * dsigy

