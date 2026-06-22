import math

def warmup_cosine_schedule(base_lr, warmup_steps, total_steps):
    """
    Returns: list of learning rates
    """
    step_counter=0
    lr = 0
    lr_list = []
    while step_counter < warmup_steps:
        lr = base_lr * (step_counter +1)/warmup_steps
        lr_list.append(lr)
        step_counter += 1
    while step_counter < total_steps:
        lr = base_lr * .5 *(1+math.cos(math.pi*((step_counter-warmup_steps)/(total_steps-warmup_steps))))
        lr_list.append(lr)
        step_counter += 1
    return lr_list