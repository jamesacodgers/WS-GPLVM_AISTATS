import torch
from typing import List

from wsgplvm import WSGPLVM
from src.data import Dataset, SpectralData

def train_bass_on_spectral_data(
        model: WSGPLVM, 
        data: List[Dataset], 
        optimizer, 
        epochs: int
    ):
    elbo_list = []
    for epoch in range(epochs):
        # Compute the loss        
        loss =  - model.elbo(data)
        if torch.isnan(loss):
            print(list(model.named_parameters()))
        elbo_list.append(-loss.item())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print the loss after every 100 epochs
        # print(model.beta)
        
        if (epoch + 1) % 50 == 0:
            print("Epoch [{}/{}], ELBO: {:.4f}".format(epoch + 1, epochs, -loss.item()))

    return elbo_list

def train_gpfa_deterministic_on_spectral_data(
        model: WSGPLVM, 
        data: List[Dataset], 
        optimizer, 
        epochs: int
    ):
    elbo_list = []
    for epoch in range(epochs):
        # Compute the loss        
        loss =  - model.training_loss(data)
        if torch.isnan(loss):
            print(list(model.named_parameters()))
        
        elbo_list.append(-loss.item())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print the loss after every 100 epochs
            
        print(list(model.named_parameters()))
        print([dataset.get_r() for dataset in data])
        
        if (epoch + 1) % 50 == 0:
            print("Epoch [{}/{}], ELBO: {:.4f}".format(epoch + 1, epochs, -loss.item()))

    return elbo_list

def lbfgs_training_loop(
    model: WSGPLVM, 
    data: List[SpectralData], 
    params,
    epochs: int
):
    optimizer = torch.optim.LBFGS(
                                params, 
                                history_size=100, 
                                max_iter=100, 
                                line_search_fn="strong_wolfe"
                                )

    def closure():
        optimizer.zero_grad()
        loss = - model.elbo(data)  # Forward pass
        if loss.isnan():
            print(list(model.named_parameters()))
        loss.backward()  # Backpropagate the gradients
        # print(model.beta)
        for p in model.parameters():
            if p.grad is not None and not p.grad.is_contiguous():
                p.grad = p.grad.contiguous()
        return loss

    loss_list = []
    
    for epoch in range(epochs):
        # inducing_T2 = (model.v_x**2).sum(axis=1)
        # active = inducing_T2< 25
        # replacement = torch.randn()
        # model.v_x = torch.nn.Parameter(model.v_x[active])


        loss_list.append(-optimizer.step(closure))
        
        print("Epoch [{}/{}], ELBO: {:.4f}".format(epoch + 1, epochs, loss_list[-1].item()))

    return loss_list


def lbfgs_training_loop_gpfa_deterministic(
    model, 
    data: List[SpectralData], 
    params,
    epochs: int
):
    optimizer = torch.optim.LBFGS(
                                params, 
                                history_size=100, 
                                max_iter=100, 
                                line_search_fn="strong_wolfe"
                                )

    def closure():
        optimizer.zero_grad()
        loss = - model.training_loss(data)  # Forward pass
        if loss.isnan():
            print(list(model.named_parameters()))
        loss.backward()  # Backpropagate the gradients
        print(list(model.named_parameters()))
        print([dataset.get_r() for dataset in data])
        return loss

    loss_list = []
    for epoch in range(epochs):
        # inducing_T2 = (model.v_x**2).sum(axis=1)
        # active = inducing_T2< 25
        # replacement = torch.randn()
        # model.v_x = torch.nn.Parameter(model.v_x[active])


        loss_list.append(-optimizer.step(closure))
        
        print("Epoch [{}/{}], ELBO: {:.4f}".format(epoch + 1, epochs, loss_list[-1]))

    return loss_list



# %%
class TestModule(torch.nn.Module):
    def __init__(self, x, y):
        super(TestModule, self).__init__()
        self.x = torch.nn.Parameter(x)
        self.y = torch.nn.Parameter(y)

    def forward(self):
        return (self.x**2).sum() + self.y**2 
# %%

x_init = torch.Tensor([1,2])
y_init = torch.Tensor([1])

model = TestModule(x_init, y_init)

loss = model()

params = list(model.parameters())


param_vec = torch.cat([param.view(-1) for param in params])
new_params = [torch.empty_like(param) for param in model.parameters()]
start = 0 
for i,p in enumerate(new_params):
    end = start + p.numel()
    param_vec

    
# %%
