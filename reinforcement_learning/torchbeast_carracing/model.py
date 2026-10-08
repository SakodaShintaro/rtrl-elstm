import torch
from torch import nn
from torch.nn import functional as F

from torchbeast.layer import RTRLQuasiLSTMlayer


# R2AC agent: IMPALA ResNet vision stem + eLSTM (QuasiLSTM) core trained by RTRL.
# Adapted from torchbeast_procgen.model.RTRLQuasiLSTMNet.
# Difference: there are no separate actor processes here (synchronous rollout
# with the learner's parameters), so the actors do not run the RTRL recursion;
# the learner carries the RTRL states from one unroll to the next instead.
class RTRLQuasiLSTMNet(nn.Module):
    def __init__(self, num_actions, hidden_size, frame_shape):
        super().__init__()

        self.num_actions = num_actions
        self.hidden_size = hidden_size

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        input_channels = frame_shape[0]
        for num_ch in [16, 32, 32]:
            self.feat_convs.append(nn.Sequential(
                nn.Conv2d(input_channels, num_ch, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ))
            input_channels = num_ch
            for resnet in [self.resnet1, self.resnet2]:
                resnet.append(nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(num_ch, num_ch, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(num_ch, num_ch, kernel_size=3, stride=1, padding=1),
                ))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        with torch.no_grad():
            vision_out_dim = self.vision(torch.zeros(1, *frame_shape)).shape[1]
        self.fc = nn.Linear(vision_out_dim, hidden_size)

        # FC output size + last reward.
        self.rnn_input_dim = self.fc.out_features + 1

        self.core = RTRLQuasiLSTMlayer(self.rnn_input_dim, hidden_size, forget_bias=0.)

        self.output_gate = nn.Linear(self.rnn_input_dim + hidden_size, hidden_size)

        self.policy = nn.Linear(hidden_size, self.num_actions)
        self.baseline = nn.Linear(hidden_size, 1)

    def vision(self, x):
        for fconv, res1, res2 in zip(self.feat_convs, self.resnet1, self.resnet2):
            x = fconv(x)
            x = x + res1(x)
            x = x + res2(x)
        return torch.flatten(F.relu(x), 1)

    def initial_rnn_state(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    # No layer dim, as expected by RTRLQuasiLSTMlayer.compute_grad_rtrl
    def initial_rtrl_state(self, batch_size, device):
        H, I = self.hidden_size, self.rnn_input_dim
        return (
            torch.zeros(batch_size, H, I, device=device),
            torch.zeros(batch_size, H, I, device=device),
            torch.zeros(batch_size, H, device=device),
            torch.zeros(batch_size, H, device=device),
            torch.zeros(batch_size, H, device=device),
            torch.zeros(batch_size, H, device=device),
        )

    # frame: (T, B, C, H, W) uint8, reward: (T, B), done: (T, B) bool,
    # where done[t] means that frame[t] is the first frame of a new episode.
    # Returns everything the RTRL pass needs; core_output is (T, B, hidden).
    def forward_core(self, frame, reward, done, rnn_state):
        T, B, *_ = frame.shape
        x = self.vision(torch.flatten(frame, 0, 1).float() / 255.0)
        x = F.relu(self.fc(x))

        clipped_reward = torch.clamp(reward, -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1).view(T, B, -1)

        core_output_list = []
        # store c(t-1), z and f to avoid recomputation in the RTRL pass
        rnn_tm1_list = []
        z_output_list = []
        f_output_list = []

        notdone = (~done).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            # Reset core state to zero whenever an episode ended.
            rnn_state = nd.view(1, -1, 1) * rnn_state
            rnn_tm1_list.append(rnn_state)
            output, z_out, f_out, rnn_state = self.core(
                input.unsqueeze(0), rnn_state, is_actor=False)
            core_output_list.append(output)
            z_output_list.append(z_out)
            f_output_list.append(f_out)

        core_output = torch.cat(core_output_list)
        rnn_tm1 = torch.cat(rnn_tm1_list)
        z_outs = torch.cat(z_output_list)
        f_outs = torch.cat(f_output_list)

        return core_input, notdone, core_output, rnn_tm1, z_outs, f_outs, rnn_state

    def heads(self, core_input, core_output):
        T, B, _ = core_output.shape
        core_input = core_input.view(T * B, -1)
        core_output = core_output.view(T * B, -1)

        # apply output gate
        gate_out = torch.sigmoid(self.output_gate(torch.cat([core_input, core_output], dim=-1)))
        gate_out = core_output * gate_out

        policy_logits = self.policy(gate_out).view(T, B, self.num_actions)
        baseline = self.baseline(gate_out).view(T, B)
        return policy_logits, baseline

    @torch.no_grad()
    def act(self, frame, reward, done, rnn_state):
        core_input, _, core_output, _, _, _, rnn_state = self.forward_core(
            frame, reward, done, rnn_state)
        policy_logits, _ = self.heads(core_input, core_output)
        T, B, _ = policy_logits.shape
        action = torch.multinomial(
            F.softmax(policy_logits.view(T * B, -1), dim=1), num_samples=1)
        return action.view(T, B), policy_logits, rnn_state

    # Accumulates the RTRL gradients of the core parameters into their .grad
    # (call core.rtrl_zero_grad() before) and returns the updated RTRL state.
    def compute_grad_rtrl(self, core_input, notdone, rnn_tm1, z_outs, f_outs, rtrl_state, top_gradient):
        for input, nd, top_grad, rnn_tm1_, z_, f_ in zip(
                core_input.unbind(), notdone.unbind(), top_gradient.unbind(),
                rnn_tm1.unbind(), z_outs.unbind(), f_outs.unbind()):
            # Reset RTRL states to zero whenever an episode ended.
            Z_state, F_state, wz_state, wf_state, bz_state, bf_state = rtrl_state
            nd_vec = nd.view(-1, 1)
            nd_mat = nd.view(-1, 1, 1)
            rtrl_state = (
                nd_mat * Z_state, nd_mat * F_state,
                nd_vec * wz_state, nd_vec * wf_state,
                nd_vec * bz_state, nd_vec * bf_state)
            rtrl_state = self.core.compute_grad_rtrl(
                input.unsqueeze(0), rtrl_state, top_grad.unsqueeze(0),
                rnn_tm1_.unsqueeze(0), z_.unsqueeze(0), f_.unsqueeze(0))
        return rtrl_state
