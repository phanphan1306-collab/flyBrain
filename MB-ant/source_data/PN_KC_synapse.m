function [S] = PN_KC_synapse(dt, spikes, S)
% Parameters

phi_S = 0.93;
tau_syn_S = 3.0; % [ms]

% call synapse function
[S] = synapse(dt, S, spikes, phi_S, tau_syn_S);

end