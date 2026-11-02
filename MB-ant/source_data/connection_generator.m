% connectivity generator.m
% generate connection matrix for PN - KC synapses and KC - EN synapses
% ask for numPN ,numKC and numEN to be defined

%% PN-KC connection: each KC receives randomly 10 PNs
connection_PN_KC = zeros(numPN, numKC);

for i_KC = 1:numKC % for each KC
    % generate a PN uniform-random selection index, and choose the first 10 of PNs
    PN_index = randperm(numPN);
    % update the connection matrix
    connection_PN_KC(PN_index(1:10), i_KC) = 1;
end

%% KC-EN connection: all to all
connection_KC_EN = ones(numKC, numEN);