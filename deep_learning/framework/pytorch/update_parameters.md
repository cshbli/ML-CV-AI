# Update the parameters of a neural netowrk

in-place self copy

```
with torch.no_grad():
    x.copy_(k)
```    

Let net be an instance of a neural network nn.Module. Then, to multiply all parameters by 0.9:

```python
state_dict = net.state_dict()

for name, param in state_dict.items():
    # Transform the parameter as required.
    transformed_param = param * 0.9

    # Update the parameter.
    param.copy_(transformed_param)
```    

If you want to only update weights instead of every parameter:

```python
state_dict = net.state_dict()

for name, param in state_dict.items():
    # Don't update if this is not a weight.
    if not "weight" in name:
        continue
    
    # Transform the parameter as required.
    transformed_param = param * 0.9

    # Update the parameter.
    param.copy_(transformed_param)
```    