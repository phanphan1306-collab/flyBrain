function [spike, t_spike_KC, v, u] = KC_neuron(dt, t, v, u, I, t_spike_KC)
%KC_NEURON Wrapper for the Izhikevich neurons to hold parameters.

C = 4;
a = 0.01;
b = -0.3;
c = -65;
d = 8;
k = 0.035;
v_r = -85;
v_t = -25;
epsilon_mean = 0;
epsilon_std = 0.05;

spike = 0;

% Reset
if v > v_t,
    v = c;
    u = u + d;
    spike = 1;
    t_spike_KC = t;
end

% Noise term
epsilon = epsilon_mean + epsilon_std * randn;

v = v + dt/2*((k*(v-v_r)*(v-v_t)-u+I+epsilon)/C);
v = v + dt/2*((k*(v-v_r)*(v-v_t)-u+I+epsilon)/C);
u = u + dt*(a*(b*(v-v_r)-u));

end