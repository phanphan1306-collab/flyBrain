function [ change ] = STDP( delta_t )
% STDP returns the amount of change in the synaptic weight depending on the
% amount of time between pre- and postsynaptic spike.

% delta_t = t_pre - t_post

% Parameters
A_plus = 1;
A_minus = -1;
tau_plus = 15; % [ms]
tau_minus = 15; % [ms]

% gpuArray ask for element wise operation:

change = 0;

    if delta_t < 0
        change = A_minus*exp(delta_t/tau_minus);
    elseif delta_t > 0
        change = A_minus*exp(-delta_t/tau_plus);
    end

end

