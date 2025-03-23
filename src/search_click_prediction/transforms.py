def hill(x, half_sat, slope):
    return x**slope / (x**slope + half_sat**slope)
