import torch
import torch.nn as nn
import torch.nn.functional as F

class S6Block(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        d_inner = int(expand*d_model)
        self.d_inner = d_inner

        self.in_proj = nn.Linear(d_model, 2*d_inner)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv-1)
        self.x_proj = nn.Linear(d_inner, d_state*2+1) #(B, C, dt)
        self.dt_proj = nn.Linear(1, d_inner)

        #A: log-parameterized for stability
        A = torch.arange(1, d_state+1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)
        nn.init.uniform_(self.dt_proj.bias, -4.0, -1.0)

    def forward(self, x):
        #x: (B, L, d_model)
        B, L, _ = x.shape
        xz = self.in_proj(x) #(B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1) #each (B, L, d_inner)

        #depthwise conv over time
        x_conv = self.conv1d(x_in.transpose(1,2))[..., :L].transpose(1,2)
        x_conv = F.silu(x_conv)

        #selective parameters
        bc_dt = self.x_proj(x_conv) #(B, L, 2*d_state+1)
        B_mat, C_mat, dt_raw = bc_dt.split([self.d_state, self.d_state, 1], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw)) #(B, L, d_inner)

        #discretize: A_bar = exp(dt*A), B_bar = dt*B
        A = -torch.exp(self.A_log) #(d_inner, d_state)

        #simple sequential recurrence
        h = x.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1) #(B, d_inner, 1)
            A_bar = torch.exp(dt_t*A.unsqueeze(0)) #(B, d_inner, d_state)
            B_t = B_mat[:, t, :].unsqueeze(1) #(B, 1, d_state)
            x_t = x_conv[:, t, :].unsqueeze(-1) #(B, d_inner, 1)
            h = A_bar*h+(dt_t*x_t)*B_t
            C_t = C_mat[:, t, :].unsqueeze(1) #(B, 1, d_state)
            y_t = (h*C_t).sum(-1) #(B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1) #(B, L, d_inner)
        y = y+self.D*x_conv
        y = y*F.silu(z)
        return self.out_proj(y)

class MambaBackbone(nn.Module):
    def __init__(self, d_model=64, n_layers=4, d_state=16):
        super().__init__()
        self.layers = nn.ModuleList([S6Block(d_model, d_state=d_state) for _ in range (n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, emb):
        #emb: (B, P, d_model)
        for layer in self.layers:
            emb = layer(emb) + emb
        return self.norm(emb)