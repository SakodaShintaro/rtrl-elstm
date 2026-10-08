# Gradient test for torchbeast_carracing: with fixed parameters, the RTRL
# gradients of the eLSTM core accumulated over two consecutive unrolls (RTRL
# state carried across them, episode reset in the middle of each) must equal
# the BPTT gradients over the concatenated sequence.
#
# Run from reinforcement_learning/: uv run python -m tests.carracing_rtrl_grad_test

import copy

import torch

from torchbeast_carracing.model import RTRLQuasiLSTMNet

def head_loss(model, core_input, core_output, w_logits, w_baseline):
    policy_logits, baseline = model.heads(core_input, core_output)
    return (policy_logits * w_logits).sum() + (baseline * w_baseline).sum()


def main():
    torch.manual_seed(0)

    T, B, A, H = 7, 3, 5, 16
    frame_shape = (3, 24, 24)
    model = RTRLQuasiLSTMNet(A, H, frame_shape)

    frames = torch.randint(0, 256, (2 * T + 1, B, *frame_shape), dtype=torch.uint8)
    rewards = torch.randn(2 * T + 1, B)
    dones = torch.zeros(2 * T + 1, B, dtype=torch.bool)
    dones[0] = True
    dones[3, 0] = True  # reset inside unroll 1
    dones[T, 1] = True  # reset exactly at the unroll boundary
    dones[T + 4, 2] = True  # reset inside unroll 2
    w_logits = torch.randn(2 * T, B, A)
    w_baseline = torch.randn(2 * T, B)

    # Reference: BPTT over frames 0..2T-1 with core parameters requiring grad.
    ref = copy.deepcopy(model)
    for param in ref.core.parameters():
        param.requires_grad_(True)
    rnn_state = ref.initial_rnn_state(B, "cpu")
    core_input, _, core_output, _, _, _, _ = ref.forward_core(
        frames[:2 * T], rewards[:2 * T], dones[:2 * T], rnn_state)
    head_loss(ref, core_input, core_output, w_logits, w_baseline).backward()

    # RTRL: two unrolls of T + 1 frames, as in train.py.
    model.zero_grad()
    model.core.rtrl_zero_grad()
    rnn_state = model.initial_rnn_state(B, "cpu")
    rtrl_state = model.initial_rtrl_state(B, "cpu")
    for u in range(2):
        s = slice(u * T, u * T + T + 1)
        core_input, notdone, core_output, rnn_tm1, z_outs, f_outs, _ = model.forward_core(
            frames[s], rewards[s], dones[s], rnn_state)
        core_output.retain_grad()
        head_loss(model, core_input[:T], core_output[:T],
                  w_logits[u * T:(u + 1) * T], w_baseline[u * T:(u + 1) * T]).backward()
        rtrl_state = model.compute_grad_rtrl(
            core_input[:T].detach(), notdone[:T], rnn_tm1[:T].detach(),
            z_outs[:T], f_outs[:T], rtrl_state, core_output.grad[:T])
        # state before frame (u + 1) * T, as held by the actor in train.py
        with torch.no_grad():
            s = slice(u * T, (u + 1) * T)
            rnn_state = model.forward_core(frames[s], rewards[s], dones[s], rnn_state)[-1]

    for (name, p_bptt), p_rtrl in zip(ref.core.named_parameters(), model.core.parameters()):
        g_bptt, g_rtrl = p_bptt.grad, p_rtrl.grad
        rel_error = ((g_bptt - g_rtrl).norm() / g_bptt.norm()).item()
        print(f"{name:>7s}: relative error {rel_error:.2e} (grad norm {g_bptt.norm().item():.3e})")
        assert rel_error < 1e-5, name
    print("All tests pass.")


if __name__ == "__main__":
    main()
