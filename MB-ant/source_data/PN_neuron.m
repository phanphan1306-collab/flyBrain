function [spike, v, u] = PN_neuron(dt, v, u, I)
%PN_NEURON Wrapper for the Izhikevich neurons to hold parameters.

C =100;
a = 0.3;
b = -0.2;
c = -65;
d =8;
k = 2;
v_r = -60;
v_t = -40;
epsilon_mean = 0;
epsilon_std = 0.05;

spike = 0;

% Reset
if v > v_t,
    v = c;
    u = u + d;
    spike = 1;
end

% Noise term
epsilon = epsilon_mean + epsilon_std * randn;

v = v + dt/2*((k*(v-v_r)*(v-v_t)-u+I+epsilon)/C);
v = v + dt/2*((k*(v-v_r)*(v-v_t)-u+I+epsilon)/C);
u = u + dt*(a*(b*(v-v_r)-u));

end

