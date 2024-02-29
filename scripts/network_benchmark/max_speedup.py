
def max_speedup(tf, tb, tc):
    #return (tf+tb+tc)/(tf+tb+tc-min(tb,tc))
    return 32*(tf+tb)/(tf+tb+tc-min(tb,tc))

def tcmin(num_gradients):
    return 2*num_gradients*32 / (9.43e9)

data = {
        'resnet152': (0.091, 0.182, 60192808),
        'densenet201': (0.073, 0.146, 20013928),
        'inceptionv4': (0.06315374, 0.06315374*2, 42679816),
        'bertbase': (0.10182975333333333, 0.10182975333333333*2, 110111042),
        }
for k in data:
    tf, tb, num_gradients = data[k]
    tc = tcmin(num_gradients)
    print('%s: %f' % (k, max_speedup(tf, tb, tc)))
print
