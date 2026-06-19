import torch

def tensor_op(x, y, op):
    """
    Returns: list (result tensor converted via .tolist())
    """
    x = torch.tensor(x)
    y = torch.tensor(y)
    if op == 'add':
        myTensor = torch.add(x,y)
        myList = myTensor.tolist()
        return myList
    if op == 'multiply':
        myTensor = torch.multiply(x,y)
        myList = myTensor.tolist()
        return myList
    if op == 'matmul':
        myTensor = torch.matmul(x,y)
        myList = myTensor.tolist()
        return myList
    if op == 'power':
        myTensor = torch.pow(x,y)
        myList = myTensor.tolist()
        return myList
    if op == 'max':
        myTensor = torch.max(x,y)
        myList = myTensor.tolist()
        return myList
    else:
        return None