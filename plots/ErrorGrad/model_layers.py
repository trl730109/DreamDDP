

def get_res18_layers():
    layers = ["conv1.weight"]
    res18_planes = [2, 2, 2, 2]
    for l in range(1, 5):
        for plane in range(res18_planes[l-1]):
            layers.append(f"layer{l}.{plane}.conv1.weight")
            layers.append(f"layer{l}.{plane}.conv2.weight")
    layers.append("linear.weight")
    return layers
























































