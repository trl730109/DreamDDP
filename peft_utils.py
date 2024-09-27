import torch

def get_peft_named_gradients(model):
    named_grads = {}
    for name, param in model.named_parameters():
        assert "lora" in name
        if param.grad is not None:
            named_grads[name] = param.grad.clone()
    return named_grads

def sum_peft_gradients(gradients_list):
    summed_grads = {}
    for name in gradients_list[0].keys():
        assert "lora" in name
        summed_grads[name] = sum(grad[name] for grad in gradients_list)
    return summed_grads

def avg_peft_gradients(gradients_list):
    avg_grads = {}
    length = len(gradients_list)
    for name in gradients_list[0].keys():
        assert "lora" in name
        avg_grads[name] = sum(grad[name] for grad in gradients_list) / length
    return avg_grads




















