import torch
from torch import nn
from torch_geometric.nn import MessagePassing, InstanceNorm

class Swish(nn.Module):
    """Swish activation function
    """
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)



class GNN_Layer(MessagePassing):
    """Message passing layer
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: int,
                 time_window: int,
                 spatial_dim: int,
                 n_variables: int):
        """Initialize message passing layers
        Args:
            in_features (int): number of node input features
            out_features (int): number of node output features
            hidden_features (int): number of hidden features
            time_window (int): number of input/output timesteps (temporal bundling)
            spatial_dim (int): number of dimension of spatial domain  
            n_variables (int): number of equation specific parameters used in the solver
        """
        super(GNN_Layer, self).__init__(node_dim=-2, aggr='mean') # node_dim: The axis along which to propagate. (default: -2)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        assert (spatial_dim == 1 or spatial_dim == 2 or spatial_dim == 3)

        self.message_net_1 = nn.Sequential(nn.Linear(2 * in_features + time_window + spatial_dim + n_variables, hidden_features), 
                                           Swish()
                                           )
        self.message_net_2 = nn.Sequential(nn.Linear(hidden_features, hidden_features), 
                                           Swish()
                                           )
        self.update_net_1 = nn.Sequential(nn.Linear(in_features + hidden_features + n_variables, hidden_features), 
                                          Swish()
                                          )
        self.update_net_2 = nn.Sequential(nn.Linear(hidden_features, out_features), 
                                          Swish()
                                          )
        self.norm = InstanceNorm(hidden_features)

    def forward(self, x, u, pos, variables, edge_index, batch):
        """Propagate messages along edges
        """
        x = self.propagate(edge_index, x=x, u=u, pos=pos, variables=variables)
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        """Message update following formula 8 of the paper
        """
        message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, variables_i), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x, variables):
        """Node update following formula 9 of the paper
        """
        update = self.update_net_1(torch.cat((x, message, variables), dim=-1))
        update = self.update_net_2(update)
        if self.in_features == self.out_features:
            return x + update
        else:
            return update


class PhysicsAwareGNNLayer(MessagePassing):
    """消息传递中显式融入多变量物理关系"""
    def __init__(self, in_features, hidden_features, out_features,
                 spatial_dim, time_window, num_physics_vars, n_eq_vars,
                 zero_var_diff=False):
        super().__init__(node_dim=-2, aggr='mean')
        self.in_features = in_features
        self.out_features = out_features
        # [E02 消融] B 变体开关：置零显式多场差分槽的取值，但保留其权重列，
        # 因此名义参数量与 C（zero_var_diff=False）严格相同。
        self.zero_var_diff = zero_var_diff
        # 消息函数输入增强：包含所有变量的邻域差分
        msg_input_dim = 2*in_features + \
                       spatial_dim + \
                       num_physics_vars * time_window + n_eq_vars
                       # 关键：加入多变量差分
        self.message_net = nn.Sequential(
            nn.Linear(msg_input_dim, hidden_features),
            Swish(),
            nn.Linear(hidden_features, hidden_features),
            Swish()
        )
        self.update_net = nn.Sequential(
            nn.Linear(in_features + hidden_features + n_eq_vars, hidden_features),
            Swish(),
            nn.Linear(hidden_features, out_features)
        )
        self.norm = InstanceNorm(hidden_features)
        
    def forward(self, x, u, pos, variables, edge_index, batch):
        """Propagate messages along edges
        """
        x = self.propagate(edge_index, x=x, u=u, pos=pos, variables=variables)
        x = self.norm(x, batch)
        return x
    
    def message(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        # u_i, u_j: [?, time_window, num_physics_vars] → 计算多变量差分
        # [E02 消融] zero_var_diff=True 时该槽恒为零（B 变体）；消息输入维度
        # 不变（仍为 msg_input_dim），message_net[0] 仍为 [hidden, msg_input_dim]，
        # 故 B 与 C 的名义参数量严格相同。
        if self.zero_var_diff:
            var_diff = torch.zeros_like(u_i)
        else:
            var_diff = (u_i - u_j) # .reshape(u_i.shape[0], -1)  # 展平所有变量差分
        msg_input = torch.cat([x_i, x_j, pos_i - pos_j, var_diff, variables_i], dim=-1)
        return self.message_net(msg_input) # (E, hidden_features)
    
    def update(self, msg, x, variables):
        # 在进入update前，自动进行聚合agg，msg形状变为(N, hidden_features)
        update = self.update_net(torch.cat([x, msg, variables], dim=-1))
        return x + update if self.in_features == self.out_features else update
