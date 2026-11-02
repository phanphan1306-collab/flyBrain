function [S] = synapse(dt, S, spikes, phi_S, tau_syn_S);
%SYNAPSE Takes as input the current "state" of the synapse and
%outputs the estimated state after time dt.

% spike indicates if a presynaptic spike occurred. If so then spike should
% equal 1, otherwise 0.
% S is the amount of neurotransmitter active at the synapse at time t
% phi_S is a quantile of the amount of neurotransmitter released when a
% presynaptic spike occurred.

% Change in S
% dSdt = -S/tau_syn_S;

% After change + Spike
S = S -S/tau_syn_S*dt + spikes*phi_S;

end

