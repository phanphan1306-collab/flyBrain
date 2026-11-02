function [S, g, c, d] = KC_EN_synapse(dt, spikes, S, g, c, delta_t, pre_post_spike_occured, d, BA )
%KC_EN_SYNAPSE Wrapper for the general synapse type to hold parameters for
%the KC-EN synapse

% tau_syn_S is the synaptic time constant.
% g is the synaptic weight or conductance (adaped in learning)
% c is the synaptic "tag"
% tau_c is a timeconstant associate with the synaptic tag.
% delta_t = t_pre - t_post
% pre_post_spike_occured indicates if a pre- or post- synaptic spike occured.
% d is an extracellular concentration of a biogenic amine
% BA is the amount of BA released.

% Parameters
tau_c = 40; % [ms]
tau_d = 20; % [ms]

phi_S = 8;
tau_syn_S = 8;
% g_max = 2.0; % maximal conductance

% call the synapse function
[S] = synapse(dt, S, spikes, phi_S, tau_syn_S);

% Change in c
dcdt = -c/tau_c;
dcdt = dcdt + pre_post_spike_occured*STDP(delta_t);


% Update c
c = c+dcdt*dt;
% Change in d
dddt = -d/tau_d;

% Update d
d = d + dddt * dt + BA;

% Change in g
dgdt = c * d;

% Update g
g = max(0.0001, g+dgdt*dt);

end