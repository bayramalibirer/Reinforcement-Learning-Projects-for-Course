#import the library and transition matrix
import torch
T= torch.tensor([[0.4, 0.6], [0.8, 0.2]])
#Calculate the transition probility after k steps. Here we will use for k=2, 5, 10, 15, 20
T_2=torch.matrix_power(T, 2)
T_5=torch.matrix_power(T, 5)
T_10=torch.matrix_power(T, 10)
T_15=torch.matrix_power(T, 15)
T_20=torch.matrix_power(T, 20)
# define the distribution of two state
v=torch.tensor([[0.7, 0.3]])
# calculate the state distribution after k= 1,2,5,10,15,20 steps
v_1=torch.mm(v, T)
v_2=torch.mm(v, T_2)
v_5=torch.mm(v, T_5)
v_10=torch.mm(v, T_10)
v_15=torch.mm(v, T_15)
v_20=torch.mm(v, T_20)

# see the output
print("Transition matrix after 2 steps : {}".format(T_2))
print("Transition matrix after 5 steps : {}".format(T_5))
print("Transition matrix after 10 steps : {}".format(T_10))
print("Transition matrix after 15 steps : {}".format(T_15))
print("Transition matrix after 20 steps : {}".format(T_20))

print("Distribution after 1 step : {}".format(v_1))
print("Distribution after 2 step : {}".format(v_2))
print("Distribution after 5 step : {}".format(v_5))
print("Distribution after 10 step : {}".format(v_10))
print("Distribution after 15 step : {}".format(v_15))
print("Distribution after 20 step : {}".format(v_20))